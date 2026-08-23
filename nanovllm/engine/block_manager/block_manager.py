from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence

from utils.profiling import nvtx_annotate

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self._peak_used_blocks = 0

    @property
    def peak_used_blocks(self) -> int:
        return self._peak_used_blocks

    def reset_peak_usage(self) -> None:
        """Reset the high-water mark to the current physical block usage."""
        self._peak_used_blocks = len(self.used_block_ids)

    def _record_peak_usage(self) -> None:
        self._peak_used_blocks = max(
            self._peak_used_blocks,
            len(self.used_block_ids),
        )

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        Compute a hash for a token block, optionally chained to a prefix hash.

        Args:
            token_ids: Token IDs in the block.
            prefix: Hash of the preceding block, or -1 for the first block.

        Returns:
            The block hash as an integer.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """Allocate a physical block from the free list and return its ID."""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # Invalidate the old hash mapping before reusing this physical block.
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self._record_peak_usage()
        return block_id

    def _deallocate_block(self, block_id: int):
        """Return an unreferenced physical block to the free list."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        Determine how many prefix blocks can be reused for a sequence.

        Returns:
            The number of reusable cached blocks, or -1 when there are not
            enough free blocks to allocate the uncached remainder.
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks

        # The last block may be incomplete, so it is not reused as a cached block.
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    @nvtx_annotate("BlockManager.allocate")
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        Allocate blocks for a sequence, reusing cached prefix blocks when possible.

        Args:
            seq: The sequence receiving a physical block table.
            num_cached_blocks: Number of cached prefix blocks to reuse.
        """
        assert not seq.block_table
        h = -1

        # Attach cached prefix blocks, sharing blocks that are already in use.
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        # Allocate fresh blocks for the uncached remainder of the sequence.
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        self._record_peak_usage()
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    @nvtx_annotate("BlockManager.deallocate")
    def deallocate(self, seq: Sequence):
        """Release every physical block associated with a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Return whether decode can append a token without exhausting blocks."""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Allocate a new block when the appended token starts one."""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """Hash newly completed blocks and add them to the prefix cache."""
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id


def create_component(*, num_blocks: int, block_size: int) -> BlockManager:
    return BlockManager(num_blocks, block_size)
