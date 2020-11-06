# =============================================================================
# Linear Agent Base Class
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np


class BaseLinearAgent(object):
    def __init__(self, feature_dim, num_actions, gamma=0.9, lr=0.1, seed=0):
        """
        TODO define arguments
        """
        self.feature_dim = feature_dim  # feature dimension
        self.num_actions = num_actions  # number of discrete actions
        self.gamma = gamma
        self.lr = lr  # step size / learning rate

        # Saving single-episode trajectory
        self.traj = None

        # RNG
        self.rng = np.random.default_rng(seed)

    def begin_episode(self, phi_0):
        """
        Start of episode
        :param observation: integer denoting tabular state index
        :return: integer action index
        """
        # Select action
        cur_act = self._select_action(phi_0)

        # Initialize trajectory
        self.traj = {
            'phi': [phi_0],
            'a': [cur_act],
            'r': []
        }

        return cur_act

    def step(self, phi_t: np.array, reward: float, done: bool) -> int:
        """
        Take step in the environment
        :param phi_t: vector of feature observation
        :param reward: scalar reward
        :param done: boolean for whether episode is finished
        :return: integer action index
        """
        pass

    def _select_action(self, phi) -> int:
        # selects action
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
