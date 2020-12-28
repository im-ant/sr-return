# =============================================================================
# Expected Eligibility Trace Algorithm from van Hasselt et al.
# https://arxiv.org/abs/2007.01839
# Linear implementation
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseLinearAgent


class ExpectedTraceAgent(BaseLinearAgent):
    def __init__(self, feature_dim,
                 num_actions,
                 gamma=0.9,
                 lamb=0.8,
                 lr=0.1,
                 seed=0):

        """
        TODO define arguments
        """
        super().__init__(feature_dim, num_actions, gamma=gamma, lr=lr, seed=seed)
        self.lamb = lamb
        self.eta = 0.0  # for now use just the full expected trace
        self.value_lr = lr
        self.et_lr = lr

        # Initialize Q function and trace
        self.Wq = np.zeros((self.feature_dim, self.num_actions))  # TODO check correct
        self.Z = np.zeros((self.feature_dim, self.num_actions))  # TODO check correct
        self.Wz = np.zeros((self.num_actions,
                            self.feature_dim,
                            self.feature_dim))  # trace params

    def begin_episode(self, phi):
        super().begin_episode(phi)
        self.log_dict = {
            'value_errors': [],
            'et_error_norms': [],
        }
        # Reset eligibility trace
        self.Z *= 0.0

    def step(self, phi_t: np.array, reward: float, done: bool) -> int:
        """
        Take step in the environment
        :param phi_t: vector of feature observation
        :param reward: scalar reward
        :param done: boolean for whether episode is finished
        :return: integer action index
        """

        # Get new action based on state
        new_act = self._select_action(phi_t)

        # Save trajectory
        if not done:
            self.traj['phi'].append(phi_t)
            self.traj['a'].append(new_act)
        self.traj['r'].append(reward)

        # ==
        # Learning (via trace)
        if len(self.traj['r']) > 0:
            self._optimize_model(done)

        return new_act

    def _optimize_model(self, done) -> None:
        # ==
        # Unpack current experience tuple, (S, A, R')
        t_idx = len(self.traj['r']) - 1
        cur_phi = self.traj['phi'][t_idx]
        cur_act = self.traj['a'][t_idx]
        rew = self.traj['r'][t_idx]

        # ==
        # Update transient trace
        act_vec = np.zeros(self.num_actions)
        act_vec[cur_act] = 1.0
        grad_Qw = np.outer(cur_phi, act_vec)

        self.Z = (self.lamb * self.gamma) * self.Z + grad_Qw

        # ==
        # Supervised learning of expected trace  # TODO make sure below is okay
        cur_et = np.transpose(self.Wz[cur_act]) @ cur_phi  # (d, )
        et_err = self.Z[:, cur_act] - cur_et  # (d, )
        d_Wz = np.transpose(np.outer(et_err, cur_phi.T))  # (d, d)

        self.Wz[cur_act] += self.et_lr * d_Wz  # update ET

        # ==
        # Update Q function
        if not done:
            nex_phi = self.traj['phi'][t_idx + 1]
            nex_act = self.traj['a'][t_idx + 1]
            nex_q = self.compute_Q_value(nex_phi, nex_act)
        else:
            nex_q = 0.0

        cur_q = self.compute_Q_value(cur_phi, cur_act)
        td_err = rew + (self.gamma * nex_q) - cur_q

        # Parameter udpates with ET
        cur_et = np.matmul(
            np.transpose(self.Wz, axes=(0, 2, 1)), cur_phi
        )  # (|A|, d, d) x (d,) -> (|A|, d)

        # Parameter updates
        del_Wq = td_err * np.transpose(cur_et)
        self.Wq = self.Wq + (self.lr * del_Wq)

        # ==
        # Logging losses
        self.log_dict['value_errors'].append(td_err)
        self.log_dict['et_error_norms'].append(
            np.linalg.norm(et_err)
        )

    def compute_Q_value(self, phi, act) -> float:
        """
        Helper function, compute the value given a state feature and action

        :param phi: state feature
        :param act: action
        :return: value, Q(phi, act)
        """
        q_vec = phi @ self.Wq
        return q_vec[act]

    def _select_action(self, phi) -> int:
        """
        Selects action
        :param phi:
        :return:
        """
        return 0  # TODO change

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    pass
