# =============================================================================
# Offline expected successor trace algorithm
#
# Author: Anthony G. Chen
# =============================================================================

import copy
from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseLinearAgent


class SFReturnAgent(BaseLinearAgent):
    def __init__(self, feature_dim,
                 num_actions,
                 gamma=0.9,
                 lamb=0.8,
                 lr=0.1,
                 seed=0):
        """
        TODO define arguments
        :param feature_dim:
        :param num_actions:
        :param gamma:
        :param lamb:
        :param lr:
        :param seed:
        """
        super().__init__(feature_dim, num_actions, gamma=gamma, lr=lr, seed=seed)

        self.lamb = lamb
        self.reward_lr = lr  # different learning rates?
        self.value_lr = lr
        self.sf_lr = lr

        # Weights
        self.Wr = np.zeros(self.feature_dim)
        self.Ws = np.zeros((self.num_actions,
                            self.feature_dim,
                            self.feature_dim))  # |A| * d * D
        self.Wv = np.zeros(self.feature_dim)

    def step(self, phi_t: np.array, reward: float, done: bool) -> int:
        # Get new action based on state
        new_act = self._select_action(phi_t)

        # Save trajectory
        if not done:
            self.traj['phi'].append(phi_t)
            self.traj['a'].append(new_act)
        self.traj['r'].append(reward)

        # ==
        # Learning

        # Reward learning
        if len(self.traj['r']) > 0:
            self._optimize_reward_fn()

        # SF learning
        if len(self.traj['phi']) > 1:
            self._optimize_successor_features()

        # Value learning
        if len(self.traj['phi']) > 0:
            self._optimize_value_fn()

        return new_act

    def _optimize_reward_fn(self) -> None:
        # Get most recent feature & reward
        t_idx = len(self.traj['r']) - 1
        cur_phi = self.traj['phi'][t_idx]
        cur_rew = self.traj['r'][t_idx]

        # Update reward function
        d_Wr = (cur_rew - np.dot(cur_phi, self.Wr)) * cur_phi
        self.Wr = self.Wr + (self.reward_lr * d_Wr)

    def _optimize_successor_features(self) -> None:
        # Get most recent features
        t_idx = len(self.traj['phi']) - 2
        cur_phi = self.traj['phi'][t_idx]
        cur_act = self.traj['a'][t_idx]
        nex_phi = self.traj['phi'][t_idx+1]
        nex_act = self.traj['a'][t_idx+1]

        # Compute SF TD errors
        cur_sf = self.Ws[cur_act] @ cur_phi
        nex_sf = self.Ws[nex_act] @ nex_phi
        sf_td_err = cur_phi + (self.lamb * self.gamma * nex_sf) - cur_sf

        d_Ws = np.outer(sf_td_err, cur_phi)

        # Update
        self.Ws[cur_act] += self.sf_lr * d_Ws
        # Maybe future TODO can use soft actions

    def _optimize_value_fn(self) -> None:
        # Get current feature
        t_idx = len(self.traj['phi']) - 1
        cur_phi = self.traj['phi'][t_idx]
        cur_act = self.traj['a'][t_idx]

        # Compute successor lambda return
        cur_sf = self.Ws[cur_act] @ cur_phi  # (d, )
        sl_G = cur_sf @ (self.Wr + (self.gamma * (1.0 - self.lamb) * self.Wv))  # scalar

        sl_err = (sl_G - (cur_phi @ self.Wv))
        d_Wv = sl_err * cur_phi

        # Update
        self.Wv += self.value_lr * d_Wv

    def _select_action(self, phi) -> int:
        return 0

    def _optimize_model(self) -> None:
        pass





# ==
# For testing purposes only
if __name__ == "__main__":
    pass
