# =============================================================================
# Offline expected successor trace algorithm
#
# Author: Anthony G. Chen
# =============================================================================

import copy
from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseAgent
from algos.exp_strace_agent import STraceAgent


class SRAgent(STraceAgent):
    def __init__(self, n_states,
                 gamma=0.9,
                 lamb=0.8,
                 lr=0.1,
                 s_prop_sample=0.10,
                 use_true_s_mat=False,
                 use_rand_s_mat=False,
                 use_true_r_fn=False,
                 seed=0):
        # TODO define more arguments
        """
        TODO define arguments
        """
        super().__init__(n_states, gamma=gamma, lamb=1.0, lr=lr,
                         s_prop_sample=s_prop_sample,
                         use_true_s_mat=use_true_s_mat,
                         use_rand_s_mat=use_rand_s_mat,
                         seed=seed)

        # ==
        # Initialize the reward function
        self.use_true_r_fn = use_true_r_fn
        self.R = np.zeros(self.n_states)
        if self.use_true_r_fn:
            # NOTE: exclusive to the random chain case
            self.R[-1] = 0.5

    def _optimize_model(self) -> None:
        # ==
        # Learning

        # Unpack trajectory and estimate values
        r_traj = self.traj['r']
        s_traj = self.traj['s']

        # Learn expected forward ("successor") trace
        if not self.use_true_s_mat:
            self._update_successor_trace_anytime(s_traj)

        # Learn the reward via supervised learning
        if not self.use_true_r_fn:
            self._update_reward_fn(s_traj, r_traj)

        # ==
        # Compute the value function
        self.V = np.dot(self.S_mat, self.R)

        # ==
        # TODO log losses?
        pass

    def _update_reward_fn(self, s_traj, r_traj):
        """
        Update the reward function via supervised learning
        :param s_traj:
        :param r_traj:
        :return:
        """
        for t in range(len(s_traj)):
            s_t, r_t = s_traj[t], r_traj[t]
            r_delta = r_t - self.R[s_t]
            self.R[s_t] = self.R[s_t] + (self.lr * r_delta)

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = STraceAgent(n_states=5)

    print(agent)
    print(agent.V)
