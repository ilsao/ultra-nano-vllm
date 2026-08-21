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

    def _pop_preemption_victim(self) -> Sequence:
        """Remove the smallest running sequence, preferring the newest tie."""
        victim = None
        victim_index = -1
        min_blocks = None
        running_count = len(self.running)
        for reverse_offset, seq in enumerate(reversed(self.running)):
            blocks = len(seq.block_table)
            if min_blocks is None or blocks < min_blocks:
                min_blocks = blocks
                victim = seq
                victim_index = running_count - reverse_offset - 1
        assert victim is not None
        del self.running[victim_index]
        return victim

    @nvtx_annotate("Scheduler.schedule")
    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Schedule sequences while scanning each prefill candidate at most once.

        Returns:
            A pair containing the scheduled sequences and a flag indicating
            whether the current phase is prefill (True) or decode (False).
        """
        scheduled_seqs = []
        num_batched_tokens = 0
        deferred: deque[Sequence] = deque()

        # During one prefill batch, remaining token capacity and free KV blocks
        # can only decrease. A rejected sequence therefore cannot become
        # feasible later in the same scan and does not need to be revisited.
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            seq = self.waiting.popleft()
            self._candidate_inspections += 1
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    deferred.append(seq)
                    continue
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_cached_blocks = 0
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Only the first scheduled prefill may be chunked. Later sequences
            # that do not fit are deferred without being inspected again.
            if scheduled_seqs and num_tokens > remaining:
                deferred.append(seq)
                continue

            if deferred:
                self._backfill_selections += 1
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                deferred.append(seq)
            scheduled_seqs.append(seq)

        deferred.extend(self.waiting)
        self.waiting = deferred

        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode one token for each selected running sequence.
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self._pop_preemption_victim())
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
        """Update generated sequences and release completed requests."""
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


def create_component(*, config: Config, block_manager) -> Scheduler:
    return Scheduler(config, block_manager)
