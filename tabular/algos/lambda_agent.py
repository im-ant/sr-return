# =============================================================================
# Offline lambda return algorithm
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseAgent


class LambdaAgent(BaseAgent):
    def __init__(self, n_states,
                 gamma=0.9,
                 lamb=0.8,
                 lr=0.1,
                 seed=0):
        # TODO define more arguments
        """
        TODO define arguments
        """
        super().__init__(n_states, gamma=gamma, lr=lr, seed=0)
        self.lamb = lamb

    def step(self, observation: int, reward: float, done: bool) -> int:
        """
        Take step in the environment
        :param observation: integer denoting tabular state index
        :param reward: scalar reward
        :param done: boolean for whether episode is finished
        :return: integer action index
        """
        # Save trajectory
        if not done:
            self.traj['s'].append(observation)
        self.traj['r'].append(reward)

        # End of episode learning
        if done:
            self._optimize_model()

        return 0

    def _optimize_model(self) -> None:
        # ==
        # Learning

        # Unpack trajectory and estimate values
        r_traj = self.traj['r']
        s_traj = self.traj['s']
        v_traj = [self.V[s] for s in s_traj]

        # Construct lambda return
        lamb_G = self.compute_lambda_return(r_traj, v_traj)

        # Compute error and learn
        deltas = lamb_G - v_traj

        # Optimize model
        # NOTE: not entirely offline this way (potential TODO change to offline)
        for t in range(len(s_traj)):
            self.V[s_traj[t]] += self.lr * deltas[t]

        # ==
        # TODO log losses?
        pass

    def compute_lambda_return(self, r_traj, v_traj):
        """
        Helper method to compute the lambda return for a single full-
        episode trajectory of length T

        :param r_traj: trajectory of return, [r_1, r_2, ..., r_T]
        :param v_traj: trajectory of value estimate [v_0, v_1, ..., v_{T-1}]
        :return:
        """
        # Initialize (NOTE only works on single traj)
        lamb_G = np.zeros(len(r_traj))

        # ==
        # Calculate the lambda return for the trajectory

        # G_{T-1} is just the final reward (only valid for full epis trajs)
        lamb_G[-1] = r_traj[-1]

        # Compute lambda return via DP
        for i in reversed(range(len(lamb_G) - 1)):
            lamb_G[i] = (r_traj[i]
                         + ((self.gamma * (1 - self.lamb)) * v_traj[i + 1])
                         + ((self.gamma * self.lamb) * lamb_G[i + 1]))

        return lamb_G

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = LambdaAgent(n_states=5)

    print(agent)
    print(agent.V)
