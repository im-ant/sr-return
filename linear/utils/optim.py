# =============================================================================
# Helper class for optimization. Idea is to have classes that generate
# modified-gradients for usual numpy matrix updates. 
#
# Author: Anthony G. Chen
# =============================================================================

import numpy as np


class SGD:
    """
    Base stochastic gradient descent class
    """

    def __init__(self, param_like, lr=0.01):
        self.param_like = param_like
        self.lr = lr

    def step(self, grad):
        """
        Simple SGD step
        :param grad: gradient of parameters
        :return: gradient step to take
        """
        return self.lr * grad


class RMSProp(SGD):
    """
    RMSProp with bias correction
    """

    def __init__(self, param_like, lr=0.01, smoothing_alpha=0.99, eps=1e-8):
        super().__init__(param_like, lr)
        self.smoothing_alpha = smoothing_alpha
        self.eps = eps

        # Initialize parameters
        self.ms_grads = np.zeros_like(self.param_like)  # mean squared grads

        self.num_steps = 0

    def step(self, grad):
        # Exponential update of mean squared gradients
        self.ms_grads = ((self.smoothing_alpha * self.ms_grads)
                         + (1 - self.smoothing_alpha) * (grad ** 2))

        # Compute the adaptive denominator for the gradient
        # NOTE: with initial bias correction
        denom = np.sqrt(
            self.ms_grads /
            (1 - (self.smoothing_alpha ** (self.num_steps + 1)))
        )

        # Compute the delta for the parameter updates
        ada_grad = self.lr * (grad / (denom + self.eps))

        # Update counter states
        self.num_steps += 1

        return ada_grad


if __name__ == "__main__":
    print('hello world')
