from collections import deque
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from nanovllm import EngineComponent
from nanovllm.engine.sequence import Sequence, SequenceStatus


class SchedulerV2Tests(unittest.TestCase):
    def make_scheduler(
        self,
        selector: str,
        *,
        num_blocks: int = 64,
        block_size: int = 2,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 8,
    ):
        config = SimpleNamespace(
            num_kvcache_blocks=num_blocks,
            kvcache_block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos=0,
        )
        return EngineComponent(scheduler=selector).create_scheduler(config)

    def test_prefill_scans_each_waiting_sequence_at_most_once(self):
        with patch.object(Sequence, "block_size", 2):
            scheduler = self.make_scheduler(
                "scheduler-v2",
                max_num_batched_tokens=8,
            )
            first = Sequence([1, 2, 3, 4])
            blocked = Sequence([5, 6, 7, 8, 9, 10, 11])
            first_backfill = Sequence([12, 13])
            second_backfill = Sequence([14, 15])
            for seq in (first, blocked, first_backfill, second_backfill):
                scheduler.add(seq)

            scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first, first_backfill, second_backfill])
        self.assertEqual(list(scheduler.waiting), [blocked])
        telemetry = scheduler.get_scheduler_metrics()
        self.assertEqual(telemetry.candidate_inspections, 4)
        self.assertEqual(telemetry.backfill_selections, 2)

    def test_prefill_matches_v1_across_randomized_workloads(self):
        rng = random.Random(0)
        with patch.object(Sequence, "block_size", 2):
            for case in range(100):
                lengths = [rng.randint(1, 14) for _ in range(rng.randint(2, 12))]
                max_tokens = rng.randint(3, 18)
                max_seqs = rng.randint(1, 8)
                v1 = self.make_scheduler(
                    "scheduler-v1",
                    max_num_seqs=max_seqs,
                    max_num_batched_tokens=max_tokens,
                )
                v2 = self.make_scheduler(
                    "scheduler-v2",
                    max_num_seqs=max_seqs,
                    max_num_batched_tokens=max_tokens,
                )
                v1_sequences = [Sequence(list(range(length))) for length in lengths]
                v2_sequences = [Sequence(list(range(length))) for length in lengths]
                for seq in v1_sequences:
                    v1.add(seq)
                for seq in v2_sequences:
                    v2.add(seq)

                v1_scheduled, v1_prefill = v1.schedule()
                v2_scheduled, v2_prefill = v2.schedule()

                with self.subTest(case=case, lengths=lengths):
                    self.assertEqual(v2_prefill, v1_prefill)
                    self.assertEqual(
                        self._indexes(v2_scheduled, v2_sequences),
                        self._indexes(v1_scheduled, v1_sequences),
                    )
                    self.assertEqual(
                        self._indexes(v2.waiting, v2_sequences),
                        self._indexes(v1.waiting, v1_sequences),
                    )
                    self.assertEqual(
                        self._indexes(v2.running, v2_sequences),
                        self._indexes(v1.running, v1_sequences),
                    )
                    self.assertEqual(
                        [seq.num_scheduled_tokens for seq in v2_sequences],
                        [seq.num_scheduled_tokens for seq in v1_sequences],
                    )
                    self.assertEqual(
                        [seq.block_table for seq in v2_sequences],
                        [seq.block_table for seq in v1_sequences],
                    )
                    self.assertEqual(
                        v2.get_scheduler_metrics().backfill_selections,
                        v1.get_scheduler_metrics().backfill_selections,
                    )

    def test_prefill_allocation_failure_matches_v1(self):
        with patch.object(Sequence, "block_size", 2):
            schedulers = [
                self.make_scheduler(
                    selector,
                    num_blocks=2,
                    max_num_batched_tokens=4,
                )
                for selector in ("scheduler-v1", "scheduler-v2")
            ]
            sequence_groups = []
            for scheduler in schedulers:
                sequences = (
                    Sequence([1, 2, 3, 4, 5, 6]),
                    Sequence([7, 8, 9, 10]),
                )
                sequence_groups.append(sequences)
                for seq in sequences:
                    scheduler.add(seq)

            outcomes = [scheduler.schedule() for scheduler in schedulers]

        self.assertEqual(
            self._indexes(outcomes[1][0], sequence_groups[1]),
            self._indexes(outcomes[0][0], sequence_groups[0]),
        )
        self.assertEqual(
            self._indexes(schedulers[1].waiting, sequence_groups[1]),
            self._indexes(schedulers[0].waiting, sequence_groups[0]),
        )

    def test_preemption_matches_v1_and_preserves_rightmost_tie(self):
        schedulers = [
            self.make_scheduler(selector, max_num_seqs=1)
            for selector in ("scheduler-v1", "scheduler-v2")
        ]
        sequence_groups = []
        for scheduler in schedulers:
            current = Sequence([1])
            oldest = Sequence([2])
            left_tie = Sequence([3])
            right_tie = Sequence([4])
            current.block_table = [0, 1, 2]
            oldest.block_table = [3, 4, 5, 6]
            left_tie.block_table = [7]
            right_tie.block_table = [8]
            sequences = (current, oldest, left_tie, right_tie)
            sequence_groups.append(sequences)
            for seq in sequences:
                seq.status = SequenceStatus.RUNNING
            scheduler.running = deque(sequences)
            scheduler.block_manager = Mock()
            scheduler.block_manager.can_append.side_effect = [False, True]

        outcomes = [scheduler.schedule() for scheduler in schedulers]

        self.assertEqual(
            self._indexes(outcomes[1][0], sequence_groups[1]),
            self._indexes(outcomes[0][0], sequence_groups[0]),
        )
        self.assertEqual(
            self._indexes(schedulers[1].waiting, sequence_groups[1]),
            self._indexes(schedulers[0].waiting, sequence_groups[0]),
        )
        self.assertEqual(
            self._indexes(schedulers[1].running, sequence_groups[1]),
            self._indexes(schedulers[0].running, sequence_groups[0]),
        )
        self.assertEqual(
            self._indexes(schedulers[1].waiting, sequence_groups[1]),
            [3],
        )

    def test_consecutive_preemptions_match_v1(self):
        schedulers = [
            self.make_scheduler(selector, max_num_seqs=1)
            for selector in ("scheduler-v1", "scheduler-v2")
        ]
        sequence_groups = []
        for scheduler in schedulers:
            current = Sequence([1])
            smallest = Sequence([2])
            next_smallest = Sequence([3])
            largest = Sequence([4])
            current.block_table = [0, 1, 2, 3]
            smallest.block_table = [4]
            next_smallest.block_table = [5, 6]
            largest.block_table = [7, 8, 9]
            sequences = (current, smallest, next_smallest, largest)
            sequence_groups.append(sequences)
            for seq in sequences:
                seq.status = SequenceStatus.RUNNING
            scheduler.running = deque(
                [current, largest, next_smallest, smallest]
            )
            scheduler.block_manager = Mock()
            scheduler.block_manager.can_append.side_effect = [False, False, True]

        outcomes = [scheduler.schedule() for scheduler in schedulers]

        self.assertEqual(
            self._indexes(outcomes[1][0], sequence_groups[1]),
            self._indexes(outcomes[0][0], sequence_groups[0]),
        )
        self.assertEqual(
            self._indexes(schedulers[1].waiting, sequence_groups[1]),
            self._indexes(schedulers[0].waiting, sequence_groups[0]),
        )
        self.assertEqual(
            self._indexes(schedulers[1].running, sequence_groups[1]),
            self._indexes(schedulers[0].running, sequence_groups[0]),
        )
        self.assertEqual(
            schedulers[1].block_manager.deallocate.call_args_list,
            [call(sequence_groups[1][1]), call(sequence_groups[1][2])],
        )

    @staticmethod
    def _indexes(sequences, reference) -> list[int]:
        return [
            next(
                index
                for index, candidate in enumerate(reference)
                if seq is candidate
            )
            for seq in sequences
        ]


if __name__ == "__main__":
    unittest.main()
