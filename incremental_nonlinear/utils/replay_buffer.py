# ============================================================================
# Simple buffer, taken from the MinAtar code base
#
# Author: Anthony G Chen
# ============================================================================

import numpy as np


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
