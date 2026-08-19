import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """
        Initialize the LLMEngine with the specified model and configuration parameters.
        If tensor_parallel_size is greater than 1, it will spawn multiple processes 
        for tensor parallelism.

        param:
            model: str
                The model name or path to the model directory.
            **kwargs: dict
                Additional configuration parameters for the model, such as:
                - tensor_parallel_size: int
                    The number of processes to spawn for tensor parallelism.
        """

        config_fields = {field.name for field in fields(Config)}
        # Filter kwargs to only include those that are valid for Config
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # Create a Config instance from the filtered parameters
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self._closed = False
        # ps stands for processes, events are used for inter-process communication
        self.ps = []
        self.events = []
        # Use 'spawn' start method for multiprocessing to avoid issues with CUDA and PyTorch
        # `spawn` will start a fresh Python interpreter process, and import the modules again
        ctx = mp.get_context("spawn")
        # Start worker processes for tensor parallelism
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        # Initialize the tokenizer only once in the main process,
        # since we don't need to process the tokenizer in the worker threads
        # The worker threads only work for tensor parallelism
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """
        Clean up the LLMEngine exactly once.

        Explicit shutdown unregisters the atexit callback so the bound method no
        longer keeps the engine alive between sequential experiment runs.
        """
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.exit)
        try:
            self.model_runner.call("exit")
        finally:
            del self.model_runner
            for process in self.ps:
                process.join()
            self.ps.clear()
            self.events.clear()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        Add a new request to the scheduler.
        It converts the prompt to token IDs if it is a string, creates a Sequence object,
        and adds it to the scheduler.

        param:
            prompt: str | list[int]
                The input prompt for the model, either as a string or a list of token IDs.
            sampling_params: SamplingParams
                The sampling parameters for generating the output, 
                such as temperature and max tokens.
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        Perform a single step of the LLMEngine, which involves scheduling sequences,
        running the model, and post-processing the results.

        return:
            outputs: list[tuple[int, list[int]]]
                A list of tuples containing the sequence ID and the generated token IDs.
                Only finished sequences are returned.
            num_tokens: int
                The number of tokens processed in this step. If positive, it indicates
                the number of tokens scheduled in the prefill phase. If negative, it
                indicates the number of sequences processed in the decode phase.
        """
        seqs, is_prefill = self.scheduler.schedule()

        # If we are in the prefill phase, count the number of scheduled tokens.
        # If we are in the decode phase, return the negative number of sequences
        # to indicate that we are decoding.
        # Why we don't have to call the sum() in the decode phase is because 
        # we only generate one token per sequence in the decode phase
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # Concatenate the generated token IDs to the sequences, 
        # update their status, and manage the block cache.
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # Only return the finished seqs
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        Generate text for a list of prompts. 

        param:
            prompts: list[str] | list[list[int]]
                A list of input prompts for the model, either as strings or lists of token IDs.
            sampling_params: SamplingParams | list[SamplingParams]
                The sampling parameters for generating the output, such as temperature and max tokens.
            use_tqdm: bool
                Whether to display a progress bar using tqdm. Default is True.

        return:
            outputs: list[dict]
                A list of dictionaries containing the generated text and token IDs for each prompt.
        """

        # set up tqdm and initialize the sampling parameters for each prompt
        pbar = tqdm(total=len(prompts), desc="Generating", 
                    dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:  # prefill phase
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:               # decode phase
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        
        # sort the outputs by sequence ID to maintain the order of prompts
        # the type of outputs change from dict to list
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
