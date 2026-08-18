from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


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
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        Computes a hash for a given list of token IDs, optionally using a prefix hash.
        
        param:
            token_ids: list[int]
                The list of token IDs to hash.
            prefix: int, optional
                An optional prefix hash to include in the computation. Defaults to -1.
        return:
            int
                The computed hash value as an integer. 
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """
        Allocates a new block from the free list.
        
        return:
            int
                The ID of the allocated block.
        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # outdate the hash_to_block_id mapping if the block is being reused
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """
        Deallocates a block and returns it to the free list.
        
        param:
            block_id: int
                The ID of the block to deallocate.
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        Returns whether the sequence can be allocated in the cache. 
        
        param:
            seq: Sequence
                The sequence to check for allocation.
        return:
            num_cached_blocks: int
                The number of blocks that can be cached for the sequence.
                Returns -1 if the sequence cannot be allocated due to insufficient free blocks.
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks

        # The last block may not be full, we don't count it as a cached block
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

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """ 
        Allocates blocks for a sequence, using cached blocks if available.
        
        param:
            seq: Sequence
                The sequence for which blocks are to be allocated.
            num_cached_blocks: int
                The number of blocks that can be cached for the sequence. 
        """
        assert not seq.block_table
        h = -1
        
        # Allocate the cached blocks
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
        
        # Allocate the remaining blocks that are not cached
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """ 
        Deallocates the blocks associated with a sequence, returning them to the free list.
        
        param:
            seq: Sequence
                The sequence whose blocks are to be deallocated. 
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """ 
        Checks if a new block can be appended to the sequence based on 
        the current number of free blocks. 
        
        param:
            seq: Sequence
                The sequence to check for appending a new block.
        return:
            bool
                True if a new block can be appended, False otherwise.
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """ 
        Appends a new block to the sequence if the current number of tokens
        in the sequence is such that a new block is needed 
        (i.e., when the number of tokens modulo the block size equals 1).
        
        param:
            seq: Sequence
                The sequence to which a new block may be appended.
        """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        Hashes the blocks of a sequence and updates the block cache accordingly.
        
        param:
            seq: Sequence
                The sequence whose blocks are to be hashed and cached.
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return

        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
