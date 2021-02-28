# ============================================================================
# Simple buffer, taken from the MinAtar code base
#
# Author: Anthony G Chen
# ============================================================================

import numpy as np
import torch.multiprocessing as mp


class SimpleReplayBuffer:
    """
    Simple replay buffer with uniform sampling
    TODO: not sure if CPU or GPU yet - seems to be referencing to GPU
          tensors
    """

    def __init__(self, buffer_size, seed=None):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []
        self.rng = np.random.default_rng(seed=seed)

    def add(self, item):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(item)
        else:
            self.buffer[self.location] = item

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        samples = self.rng.choice(self.buffer, size=batch_size,
                                  replace=False)
        return samples

    def get_size(self):
        return len(self.buffer)


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
            batch_data = replay.sample(self.batch_size)  # (batch_size, 5) tens
            for cache_x, x in zip(cache[cache_idx], batch_data):
                for i in range(len(cache_x)):
                    cache_x[i].copy_(x[i])

        def set_up_cache():
            """Set up the memory allocation for the caches"""
            batch_data = replay.sample(self.batch_size)
            for i in range(2):
                cache.append([
                    [x.detach().clone() for x in xiter]
                    for xiter in batch_data
                ])
                for xiter in cache[i]:
                    for x in xiter:
                        x.share_memory_()
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
        item = [x.clone() for x in item]
        self.pipe.send([self.ADD, item])

    def sample(self):
        self.pipe.send([self.SAMPLE, None])
        cache_id, data = self.pipe.recv()
        if cache_id is None:
            # NOTE: didn't build in failsafe for none samples yet
            return None
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

    def get_size(self):
        self.pipe.send([self.SIZE, None])
        cur_size, __ = self.pipe.recv()
        return cur_size

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()
