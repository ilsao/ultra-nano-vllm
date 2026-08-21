from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from nanovllm import EngineComponent
from nanovllm.engine.sequence import Sequence, SequenceStatus


class OptimizedSchedulerTests(unittest.TestCase):
    def make_scheduler(
        self,
        *,
        num_blocks: int = 16,
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
        return EngineComponent(
            scheduler="scheduler-optimized"
        ).create_scheduler(config)

    def test_prefill_backfills_with_later_sequence_that_fits(self):
        with patch.object(Sequence, "block_size", 2):
            scheduler = self.make_scheduler(max_num_batched_tokens=6)
            first = Sequence([1, 2, 3, 4])
            blocked = Sequence([5, 6, 7])
            backfill = Sequence([8, 9])
            for seq in (first, blocked, backfill):
                scheduler.add(seq)

            scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first, backfill])
        self.assertEqual(sum(seq.num_scheduled_tokens for seq in scheduled), 6)
        self.assertEqual(list(scheduler.waiting), [blocked])
        self.assertEqual(list(scheduler.running), [first, backfill])
        telemetry = scheduler.get_scheduler_metrics()
        self.assertEqual(telemetry.candidate_inspections, 3)
        self.assertEqual(telemetry.backfill_selections, 1)

    def test_prefill_preserves_waiting_order_when_no_backfill_fits(self):
        with patch.object(Sequence, "block_size", 2):
            scheduler = self.make_scheduler(max_num_batched_tokens=5)
            first = Sequence([1, 2, 3, 4])
            second = Sequence([5, 6])
            third = Sequence([7, 8, 9])
            for seq in (first, second, third):
                scheduler.add(seq)

            scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first])
        self.assertEqual(list(scheduler.waiting), [second, third])

    def test_prefill_skips_head_that_cannot_allocate(self):
        with patch.object(Sequence, "block_size", 2):
            scheduler = self.make_scheduler(
                num_blocks=2,
                max_num_batched_tokens=4,
            )
            too_large = Sequence([1, 2, 3, 4, 5, 6])
            smaller = Sequence([7, 8, 9, 10])
            scheduler.add(too_large)
            scheduler.add(smaller)

            scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [smaller])
        self.assertEqual(list(scheduler.waiting), [too_large])

    def test_only_first_prefill_sequence_may_be_chunked(self):
        with patch.object(Sequence, "block_size", 2):
            scheduler = self.make_scheduler(max_num_batched_tokens=5)
            first = Sequence([1, 2, 3])
            too_large = Sequence([4, 5, 6])
            exact_fit = Sequence([7, 8])
            for seq in (first, too_large, exact_fit):
                scheduler.add(seq)

            scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first, exact_fit])
        self.assertEqual(first.num_scheduled_tokens, 3)
        self.assertEqual(exact_fit.num_scheduled_tokens, 2)
        self.assertEqual(too_large.num_scheduled_tokens, 0)
        self.assertEqual(list(scheduler.waiting), [too_large])

    def test_preemption_selects_fewest_blocks_and_schedules_current(self):
        scheduler = self.make_scheduler(max_num_seqs=1)
        current = Sequence([1])
        oldest = Sequence([2])
        smallest = Sequence([3])
        newest = Sequence([4])
        current.block_table = [0, 1, 2]
        oldest.block_table = [3, 4, 5, 6]
        smallest.block_table = [7]
        newest.block_table = [8, 9]
        for seq in (current, oldest, smallest, newest):
            seq.status = SequenceStatus.RUNNING
        scheduler.running = deque([current, oldest, smallest, newest])
        scheduler.block_manager = Mock()
        scheduler.block_manager.can_append.side_effect = [False, True]

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [current])
        self.assertEqual(list(scheduler.waiting), [smallest])
        self.assertEqual(list(scheduler.running), [current, oldest, newest])
        scheduler.block_manager.deallocate.assert_called_once_with(smallest)
        scheduler.block_manager.may_append.assert_called_once_with(current)
        telemetry = scheduler.get_scheduler_metrics()
        self.assertEqual(telemetry.preemptions, 1)
        self.assertEqual(telemetry.preempted_blocks, 1)
        scheduler.reset_scheduler_metrics()
        self.assertEqual(scheduler.get_scheduler_metrics().preemptions, 0)

    def test_preemption_tie_prefers_rightmost_sequence(self):
        scheduler = self.make_scheduler(max_num_seqs=1)
        current = Sequence([1])
        left = Sequence([2])
        right = Sequence([3])
        current.block_table = [0, 1]
        left.block_table = [2]
        right.block_table = [3]
        for seq in (current, left, right):
            seq.status = SequenceStatus.RUNNING
        scheduler.running = deque([current, left, right])
        scheduler.block_manager = Mock()
        scheduler.block_manager.can_append.side_effect = [False, True]

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [current])
        self.assertEqual(list(scheduler.waiting), [right])
        self.assertEqual(list(scheduler.running), [current, left])
        scheduler.block_manager.deallocate.assert_called_once_with(right)

    def test_preemption_retries_when_one_victim_is_not_enough(self):
        scheduler = self.make_scheduler(max_num_seqs=1)
        current = Sequence([1])
        smallest = Sequence([2])
        next_smallest = Sequence([3])
        largest = Sequence([4])
        current.block_table = [0, 1, 2, 3]
        smallest.block_table = [4]
        next_smallest.block_table = [5, 6]
        largest.block_table = [7, 8, 9]
        for seq in (current, smallest, next_smallest, largest):
            seq.status = SequenceStatus.RUNNING
        scheduler.running = deque(
            [current, largest, next_smallest, smallest]
        )
        scheduler.block_manager = Mock()
        scheduler.block_manager.can_append.side_effect = [False, False, True]

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [current])
        self.assertEqual(
            list(scheduler.waiting),
            [next_smallest, smallest],
        )
        self.assertEqual(list(scheduler.running), [current, largest])
        self.assertEqual(
            scheduler.block_manager.deallocate.call_args_list,
            [call(smallest), call(next_smallest)],
        )


if __name__ == "__main__":
    unittest.main()
