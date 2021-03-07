# ============================================================================
# Simple buffer, taken from the MinAtar code base
#
# Author: Anthony G Chen
# ============================================================================

from collections import namedtuple

import numpy as np
import torch
import torch.multiprocessing as mp


class SimpleReplayBuffer:
    """
    Simple replay buffer with uniform sampling
    """

    def __init__(self, buffer_size, seed=None):
        self.buffer_size = buffer_size
        self.location = 0
        self.cur_size = 0
        self.num_item_types = 0
        self.buffer = []
        self.rng = np.random.default_rng(seed=seed)

    def add(self, item):
        """
        Add a new item to buffer
        :param item: Iterable (list) of torch.tensors to store.
        """
        # ==
        # On first add, initialize the whole buffer with this structure
        if len(self.buffer) == 0:
            for x in item:
                # Construct repeat tuple to pre-alloc memory
                rep_sizes = [1] * (len(x.size())+1)
                rep_sizes[0] = self.buffer_size  # (buffer_size, *)

                self.buffer.append(x.repeat(rep_sizes))
            self.num_item_types = len(item)

        # ==
        # Storage and increment
        for k_i, x in enumerate(item):
            self.buffer[k_i][self.location].copy_(x)

        self.location = (self.location + 1) % self.buffer_size
        self.cur_size = min(self.buffer_size, self.cur_size + 1)

    def sample(self, batch_size):
        """
        Sample minibatch from the buffer. ASSUMES: each item type is stored
        in tensor shape (buffer_size, batch_dim=1, *) so the 2nd dim is
        squeezed.

        :param batch_size: minibatch size
        :return: samples of length self.num_item_types, each entry is a
                 torch.tensor of size (batch_size, *)
        """
        samples = []
        idxs = self.rng.integers(self.cur_size, size=batch_size)
        # NOTE TODO: potentially use choice(self.cur_size, size=batch_size, replace=False)

        for k_i in range(self.num_item_types):
            type_batch = self.buffer[k_i][idxs]  # (batch, 1, *)
            type_batch = type_batch.squeeze(1)  # (batch, *)
            samples.append(type_batch)

        return samples

    def get_size(self):
        return self.cur_size


class AsynchSimpleReplayBuffer(mp.Process):
    ADD = 0
    SAMPLE = 1
    EXIT = 2
    SIZE = 3

    def __init__(self, buffer_size, batch_size, seed=None):
        mp.Process.__init__(self)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        # TODO: make this class into a wrapper?

        self.location = 0
        self.buffer = []
        self.rng = np.random.default_rng(seed=seed)

        self.cache_len = 2

        self.pipe, self.worker_pipe = mp.Pipe()
        self.start()

    def run(self):
        replay = SimpleReplayBuffer(buffer_size=self.buffer_size,
                                    seed=self.seed)

        cache = []
        cache_initialized = False
        cur_cache_idx = 0

        def sample(cache_idx):
            """Helper for sampling buffer to cache"""
            bsamples = replay.sample(self.batch_size)  # (5, batch_size, *)
            for cache_x, x in zip(cache[cache_idx], bsamples):
                cache_x.copy_(x)

        def set_up_cache():
            """Set up the memory allocation for the caches"""
            bsamples = replay.sample(self.batch_size)  # (5, batch_size, *)
            for i in range(2):  # iterate over cache
                cache.append(bsamples)
                for x in cache[i]:
                    x.share_memory_()  # x is tensor shape (batch_size, *)
            sample(0)
            sample(1)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.ADD:
                replay.add(data)
            elif op == self.SAMPLE:
                if cache_initialized:
                    self.worker_pipe.send([cur_cache_idx, None])
                else:
                    set_up_cache()
                    cache_initialized = True
                    self.worker_pipe.send([cur_cache_idx, cache])
                cur_cache_idx = (cur_cache_idx + 1) % 2
                sample(cur_cache_idx)
            elif op == self.SIZE:
                cur_size = replay.get_size()
                self.worker_pipe.send([cur_size, None])
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def add(self, item):
        self.pipe.send([self.ADD, item])

    def sample(self):
        self.pipe.send([self.SAMPLE, None])
        cache_id, data = self.pipe.recv()
        if cache_id is None:
            # NOTE: didn't build in failsafe for none samples yet
            return None
        if data is not None:
            self.cache = data
        samples = self.cache[cache_id]
        return samples

    def get_size(self):
        self.pipe.send([self.SIZE, None])
        cur_size, __ = self.pipe.recv()
        return cur_size

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()
