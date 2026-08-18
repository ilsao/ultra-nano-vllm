import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """ 
        Initializes the ModelRunner with the given configuration, rank, and event(s).
        Prepares the model, allocates the key-value cache, 
        and sets up the environment for distributed training if applicable.
        
        param:
            config: Config
                The configuration object containing model and training parameters.
            rank: int
                The rank of the current process in a distributed training setup.
            event: Event | list[Event]
                A synchronization event or a list of events for inter-process communication.
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        # Move model to GPU
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # Beside the model, the other tensors are allocated on CPU.
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """ 
        Loops indefinitely, waiting for method calls from the main process via shared memory.
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """ 
        Waits for a method call from the main process via shared memory 
        and returns the method name and arguments.
        
        return:
            tuple
                A tuple containing the method name (str) and a list of arguments (list).
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """ 
        Writes a method call to shared memory for the worker processes to read.
        This method calulates the size of the data to be written to shared memory, 
        serializes the method name and arguments,
        and writes them to shared memory for the worker processes to read.
        
        param:
            method_name: str
                The name of the method to be called on the worker processes.
            args: list
                A list of arguments to be passed to the method being called.
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """ 
        Calls a method on the worker processes.
        
        param:
            method_name: str
                The name of the method to be called on the worker processes.
            args: list
                A list of arguments to be passed to the method being called.
        return:
            The return value of the method being called.
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """ 
        Warmup the model and record the peak memory usage 
        by running a dummy input through it to initialize weights and caches.`
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """ 
        Preallocates the entire key-value cache as one contiguous tensor on the GPU.
        It calculates the number of KV-cache blocks that can fit in the allowed GPU memory,
        and assigns each attention layer a view into its KV-cache slice.
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Number of KV heads handled by this tensor-parallel rank.
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # Determine the dimension of each attention head.
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads
        )

        # Memory required by one KV-cache block across all layers.
        # Factor 2 accounts for both K and V caches.
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )

        # Compute how many KV-cache blocks can fit in the allowed GPU memory.
        # The - (peak-current) term preserves the memory for the possible peak usage of the model
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization
            - used
            - peak
            + current
        ) // block_bytes

        assert config.num_kvcache_blocks > 0

        # Preallocate the entire KV cache as one contiguous tensor.
        # Shape: [K/V, layers, blocks, block_size, kv_heads, head_dim]
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        # Assign each attention layer a view into its KV-cache slice.
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """ 
        Prepares the block tables for a list of sequences, 
        padding them to the same length and moving them to GPU memory.
        Note that this method is only needed when cached prefix tokens are reused,
        as the attention kernel needs block tables to locate those cached K/V entries.

        param:
            seqs: list[Sequence]
                A list of Sequence objects for which block tables are to be prepared.
        return:
            torch.Tensor
                A tensor containing the padded block tables of the sequences,
                moved to GPU memory for efficient processing.
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        # pad the block tables to the same length and move them to GPU memory
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, 
                                    dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """ 
        Prepares the input token IDs, positions, and other necessary tensors 
        for the prefill phase of the model.
        
        param:
            seqs: list[Sequence]
                A list of Sequence objects for which the prefill tensors are to be prepared.
        return:
            tuple
                A tuple containing the input token IDs and positions,
                both moved to GPU memory.
        """
        input_ids = []  # The flattened input token IDs
        positions = []  # The flattened position indices for the input tokens

        # Cumulative sequence lengths used by variable-length attention kernels.
        # They describe where each sequence starts/ends in the flattened Q/K tensors.
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]

        # Maximum Q/K sequence lengths in this batch.
        # FlashAttention uses these values to configure the attention kernel.
        max_seqlen_q = 0
        max_seqlen_k = 0

        # Maps each newly computed token to its physical slot in the paged KV cache.
        slot_mapping = []

        # Only needed when cached prefix tokens are reused.
        block_tables = None

        for seq in seqs:
            # Tokens before start are already available in the KV cache.
            start = seq.num_cached_tokens
            
            # The scheduled tokens are the new tokens that will be computed in this prefill step. 
            # seqlen_q is the new query length. 
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q

            # KVs that are visible to the new query tokens 
            # include both:
            #   1. cached prefix tokens [0, start)
            #   2. newly scheduled tokens [start, end)
            seqlen_k = end

            # Append only the newly scheduled tokens to the flattened input.
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))

            # Record cumulative Q lengths in the flattened query tensor.
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)

            # Record cumulative K/V lengths in the logical attention input.
            # This may be larger than Q when prefix caching is used.
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # During warmup, no KV-cache blocks have been assigned yet.
            if not seq.block_table:
                continue

            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size

            for i in range(start_block, end_block):
                # Convert the logical block index into the physical KV-cache slot range.
                slot_start = seq.block_table[i] * self.block_size

                # The first block may already contain cached prefix tokens.
                if i == start_block:
                    # Calculate the offset
                    slot_start += start % self.block_size

                # Full intermediate blocks use the entire block.
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # The last block may only be partially filled.
                    slot_end = (
                        seq.block_table[i] * self.block_size
                        + end - i * self.block_size
                    )

                # Each new token gets one physical KV-cache slot.
                slot_mapping.extend(range(slot_start, slot_end))

        # If total K length is larger than total Q length, some K/V tokens come
        # from an already cached prefix. The attention kernel therefore needs
        # block tables to locate those cached K/V entries.
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # Build tensors in pinned CPU memory first, then asynchronously copy them
        # to GPU. `non_blocking=True` can overlap H2D transfer with other work.
        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)

        positions = torch.tensor(
            positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)

        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        # Store all attention metadata in the global/current execution context
        # so the attention layers can consume it during the forward pass.
        # The attention kernel uses this to store and retrieve KV cache.
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """ 
        Prepares the input token IDs and positions for the decode phase of the model.
        
        param:
            seqs: list[Sequence]
                A list of Sequence objects for which the decode tensors are to be prepared.
        return:
            tuple
                A tuple containing the input token IDs and positions,
                both moved to GPU memory.
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)

        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            False, 
            slot_mapping=slot_mapping, 
            context_lens=context_lens, 
            block_tables=block_tables
        )

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """ 
        Prepares the temperature tensor for sampling during the model's output generation.
        
        param:
            seqs: list[Sequence]
                A list of Sequence objects for which the temperature tensor is to be prepared.
        return:
            torch.Tensor
                A tensor containing the temperatures for each sequence,
                moved to GPU memory for efficient sampling.
        """
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        Run the model on the given input token IDs and positions, and return the logits.

        param:
            input_ids: torch.Tensor
                A tensor containing the input token IDs for the model.
            positions: torch.Tensor
                A tensor containing the position indices for the input tokens.
            is_prefill: bool
                A boolean flag indicating whether the current phase is prefill (True) 
                or decode (False).

        return:
            torch.Tensor
                If is_prefill is True, it returns the concatenated logits of all tokens. 
                If is_prefill is False, it returns the logits of the newly generated token.
        """
        # For prefill, when eager execution is enforced, or when the batch size
        # is too large for the captured CUDA graphs, run the model eagerly.
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()

            # Find the first graph whose batch size is >= current batch size.
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        Run the model on the given sequences and return the generated token IDs.
        
        param:
            seqs: list[Sequence]
                A list of Sequence objects representing the input sequences to be processed.
            is_prefill: bool
                A boolean flag indicating whether the current phase is prefill (True) 
                or decode (False).
        return:
            list[int]
                A list of token IDs generated by the model for the input sequences.
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """ 
        Runs the model with different batch sizes and captures CUDA graphs for efficient execution.
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # list of batch sizes for which we will capture CUDA graphs.
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], 
                        context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])           # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])       # capture

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph

            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
