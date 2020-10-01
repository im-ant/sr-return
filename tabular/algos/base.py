# =============================================================================
# Tabular Agent Base Class
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np


class BaseAgent(object):
    def __init__(self, n_states, gamma=0.9, lr=0.1, seed=0):
        # TODO define more arguments
        """
        TODO define arguments
        """
        self.n_states = n_states
        self.gamma = gamma
        self.lr = lr  # step size / learning rate

        # ==
        # Init value fn (at 0.5 according to S&B Ex 6.2?)
        self.V = np.ones(self.n_states) * 0.5

        # Saving single-episode trajectory
        self.traj = None

        # RNG
        self.rng = np.random.default_rng(seed)

    def begin_episode(self, observation: int) -> int:
        """
        Start of episode
        :param observation: integer denoting tabular state index
        :return: integer action index
        """
        # Initialize trajectory
        self.traj = {
            's': [observation],
            'r': []
        }

        return 0

    def step(self, observation: int, reward: float, done: bool) -> int:
        """
        Take step in the environment
        :param observation: integer denoting tabular state index
        :param reward: scalar reward
        :param done: boolean for whether episode is finished
        :return: integer action index
        """
        pass

    def _optimize_model(self) -> None:
        # Optimize
        # Log losses
        pass

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = BaseAgent()
    print(agent)
