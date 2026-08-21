from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler_metrics import SchedulerPolicyMetrics

from utils.profiling import nvtx_annotate

class Scheduler:
    def __init__(self, config: Config, block_manager):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.reset_scheduler_metrics()

    def reset_kvcache_metrics(self) -> None:
        self.block_manager.reset_peak_usage()

    def get_peak_kvcache_blocks(self) -> int:
        return self.block_manager.peak_used_blocks

    def reset_scheduler_metrics(self) -> None:
        self._candidate_inspections = 0
        self._backfill_selections = 0
        self._preemptions = 0
        self._preempted_blocks = 0

    def get_scheduler_metrics(self) -> SchedulerPolicyMetrics:
        return SchedulerPolicyMetrics(
            candidate_inspections=self._candidate_inspections,
            backfill_selections=self._backfill_selections,
            preemptions=self._preemptions,
            preempted_blocks=self._preempted_blocks,
        )

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    @nvtx_annotate("Scheduler.schedule")
    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Schedule sequences for processing.

        This method manages block allocation and preemption while ensuring the
        number of scheduled tokens does not exceed the current batch limit.

        Returns:
            A pair containing the scheduled sequences and a flag indicating
            whether the current phase is prefill (True) or decode (False).
        """
        scheduled_seqs = []

        # The number of tokens scheduled in the current batch.
        num_batched_tokens = 0

        # Prefill first. 
        # It's possible to improve the efficiency by adapting decode-maximal batching

        # Prefill sequences whose KV cache is not complete yet.
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # Classical FCFS, may cause efficiency issues
            self._candidate_inspections += 1
            seq = self.waiting[0]
            # The number of tokens that can still be scheduled in this batch.
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # Calculate the number of uncached tokens remaining for this sequence.
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Only the first sequence in the waiting queue may use chunked
            # prefill; otherwise this batch waits for the next scheduling step.
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # Move a sequence to the running queue once prefill is complete.
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode one token for each selected running sequence.
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # If a new block cannot be appended, preempt another sequence to
            # release its blocks. The current sequence is the final fallback.
            while not self.block_manager.can_append(seq):
                # Naive preemption, have room for improvement. 
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    @nvtx_annotate("Scheduler.preempt")
    def preempt(self, seq: Sequence):
        """Move a running sequence back to the waiting queue."""
        self._preemptions += 1
        self._preempted_blocks += len(seq.block_table)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    @nvtx_annotate("Scheduler.postprocess")
    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ):
        """
        Update sequences with generated tokens and release completed requests.

        Args:
            seqs: The sequences processed by the latest model step.
            token_ids: The token generated for each sequence.
            is_prefill: Whether the latest model step was a prefill step.
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # Skip token post-processing while chunked prefill is still active.
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # Append the token produced after prefill or during decode.
            seq.append_token(token_id)

            # Finish on EOS unless it is ignored, or at the completion limit.
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


def create_component(*, config: Config, block_manager) -> Scheduler:
    return Scheduler(config, block_manager)
