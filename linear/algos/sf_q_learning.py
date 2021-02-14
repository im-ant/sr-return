# =============================================================================
# Linear successor lambda return agent
#
# Author: Anthony G. Chen
# =============================================================================

import copy
from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseLinearAgent
from utils.optim import RMSProp


class LambdaSFQAgent(BaseLinearAgent):
    def __init__(self, feature_dim,
                 num_actions,
                 gamma=0.9,
                 lamb=0.0,
                 eta_trace=0.0,
                 lr=0.1,
                 reward_lr=None,
                 sf_lr=None,
                 policy_epsilon=0.3,
                 optim_kwargs=None,
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
        super().__init__(feature_dim, num_actions, gamma=gamma, lr=lr,
                         seed=seed)

        self.lamb = lamb
        self.eta_trace = eta_trace  # value fn bwd trace
        self.policy_epsilon = policy_epsilon

        self.value_lr = lr
        self.reward_lr = lr if reward_lr is None else reward_lr
        self.sf_lr = lr if sf_lr is None else sf_lr

        # ==
        # Initialize parameters
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs

        # Reward parameters
        self.Wr = self.rng.uniform(0.0, 1e-5,
                                   size=self.feature_dim)
        self.Wr_optim = RMSProp(self.Wr, lr=self.reward_lr, **optim_kwargs)

        # SF parameters
        self.Ws = np.zeros(
            (self.num_actions, self.feature_dim, self.feature_dim)
        )  # |A| * d * D
        ws_idxs = np.arange(self.feature_dim)
        self.Ws[:, ws_idxs, ws_idxs] = 1.0  # identity initialization
        self.Ws_optim = RMSProp(self.Ws, lr=self.sf_lr, **optim_kwargs)

        # Value parameters
        self.Wq = self.rng.uniform(
            0.0, 1e-5,
            size=(self.num_actions, self.feature_dim)
        )
        self.Wq_optim = RMSProp(self.Wq, lr=self.value_lr, **optim_kwargs)



        # ==
        # Trace  # TODO not tested for validity
        self.Zq = np.zeros((self.num_actions, self.feature_dim))

    def begin_episode(self, phi):
        action = super().begin_episode(phi)

        self.log_dict = {
            'reward_errors': [],
            'sf_error_norms': [],
            'value_errors': [],
        }

        # Reset trace  TODO need to implement and check
        self.Zq *= 0.0

        return action

    def step(self, phi_t: np.array, reward: float, done: bool) -> int:
        """
        Take step in the environment
        :param phi_t:
        :param reward:
        :param done:
        :return:
        """
        # Get new action and storage
        new_act = self._select_action(phi_t)
        if not done:
            self.traj['phi'].append(phi_t)
            self.traj['a'].append(new_act)
        self.traj['r'].append(reward)

        # ==
        # Learning
        if len(self.traj['r']) > 0:
            self._optimize_reward_fn()  # reward
            self._optimize_successor_features(done)  # sf
            self._optimize_value_fn(done)  # value fn

        return new_act

    def _optimize_reward_fn(self) -> None:
        # NOTE: learn mapping phi_{t} ->r_{t+1}
        # Get most recent features and reward
        t_idx = len(self.traj['r']) - 1
        cur_phi = self.traj['phi'][t_idx]
        cur_rew = self.traj['r'][t_idx]

        # Update reward function
        rew_err = cur_rew - np.dot(cur_phi, self.Wr)
        d_Wr = rew_err * cur_phi
        ada_d_Wr = self.Wr_optim.step(d_Wr)
        self.Wr = self.Wr + ada_d_Wr

        # (Log) Reward error
        self.log_dict['reward_errors'].append(d_Wr)

    def _optimize_successor_features(self, done) -> None:
        # Get current experience tuple (S, A)
        t_idx = len(self.traj['r']) - 1
        cur_phi = self.traj['phi'][t_idx]
        cur_act = self.traj['a'][t_idx]

        # Get next experience tuple if present (S', A')
        if not done:
            nex_phi = self.traj['phi'][t_idx + 1]
            q_vec = self.compute_lambda_Q_function(nex_phi)
            nex_act = np.argmax(q_vec)
            nex_sf = np.transpose(self.Ws[nex_act]) @ nex_phi
        else:
            nex_sf = 0.0

        # Compute SF TD errors
        cur_sf = np.transpose(self.Ws[cur_act]) @ cur_phi  # (d, )
        sf_td_err = cur_phi + (self.lamb * self.gamma * nex_sf) - cur_sf  # (d, )
        d_Ws = np.transpose(np.outer(sf_td_err, cur_phi))

        # Update  NOTE future: can use soft actions?
        del_Ws = np.zeros_like(self.Ws)
        del_Ws[cur_act] = d_Ws
        ada_del_Ws = self.Ws_optim.step(del_Ws)
        self.Ws = self.Ws + ada_del_Ws

        # (Log) Norm of the SF error vector
        self.log_dict['sf_error_norms'].append(
            np.linalg.norm(sf_td_err)
        )

    def _optimize_value_fn(self, done) -> None:
        # ==
        # Unpack current feature
        t_idx = len(self.traj['r']) - 1
        cur_phi = self.traj['phi'][t_idx]
        cur_act = self.traj['a'][t_idx]
        nex_rew = self.traj['r'][t_idx]

        # ==
        # Update trace TODO: need to test for validity
        act_vec = np.zeros(self.num_actions)
        act_vec[cur_act] = 1.0  # (num_actions, )
        grad_Wq = np.outer(act_vec, cur_phi)  # (num_actions, feature_dim)

        self.Zq = (self.lamb * self.gamma) * self.Zq + grad_Wq

        # ==
        # Compute the lambda value function return
        if not done:
            nex_phi = self.traj['phi'][t_idx + 1]
            q_vec = self.compute_lambda_Q_function(nex_phi)
            nex_q = np.max(q_vec)
        else:
            nex_q = 0.0

        # ==
        # TD learning using the SF bootstrap value
        cur_q = cur_phi.T @ self.Wq[cur_act]
        td_err = nex_rew + (self.gamma * nex_q) - cur_q

        # Parameter updates
        del_Wq = td_err * self.Zq  # (num_actions, feature_dim)
        ada_del_Wq = self.Wq_optim.step(del_Wq)
        self.Wq = self.Wq + ada_del_Wq

        # (Log) Value function error
        self.log_dict['value_errors'].append(td_err)

    def compute_lambda_Q_function(self, phi):
        """
        Helper function to compute the lambda-Q estimate
        :param phi: feature, size (featuer_dim, )
        :return: Q estimate, size (num_actions, )
        """

        sf = np.matmul(
            np.transpose(self.Ws, (0, 2, 1)),  # transpose feature dims
            phi
        )  # (num_actions, feature_dim)

        sf_theta = np.sum((sf * self.Wq), axis=1)  # (num_actions, )
        sf_w = np.matmul(sf, self.Wr)  # (num_actions, )

        q_vec = (1-self.lamb) * sf_theta + (self.lamb * sf_w)

        return q_vec

    def _select_action(self, phi) -> int:
        """
        Selects action with simple epsilon-greedy policy
        :param phi:
        :return:
        """
        rand_eps = self.policy_epsilon

        # Random action with prob epsilon
        if self.rng.binomial(n=1, p=rand_eps) == 1:
            action = self.rng.choice(self.num_actions)
        # Greedy action
        else:
            q_vec = self.Wq @ phi  # (n_actions, )
            action = np.argmax(q_vec)

        return action


# ==
# For testing purposes only
if __name__ == "__main__":
    pass
