# =============================================================================
# SARSA(lambda) algorithm with eligibility traces from Sutton & Barto. Adapted
# from chapter 12.2 TD(lambda) and SARSA(lambda)
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseLinearAgent


class SarsaLambdaAgent(BaseLinearAgent):
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

        # Initialize Q function and trace
        self.Wq = np.zeros((self.feature_dim, self.num_actions))  # TODO check correct
        self.Z = np.zeros((self.feature_dim, self.num_actions))  # TODO check correct

    def begin_episode(self, phi):
        super().begin_episode(phi)
        self.log_dict = {
            'value_errors': [],
        }

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
        if len(self.traj['phi']) > 1:
            self._optimize_model()

        return new_act

    def _optimize_model(self) -> None:
        # Unpack experience tuple (S, A, R', S', A')
        t_idx = len(self.traj['phi']) - 2
        cur_phi = self.traj['phi'][t_idx]
        cur_act = self.traj['a'][t_idx]
        rew = self.traj['r'][t_idx]
        nex_phi = self.traj['phi'][t_idx+1]
        nex_act = self.traj['a'][t_idx+1]

        # ==
        # Update trace
        act_vec = np.zeros(self.num_actions)
        act_vec[cur_act] = 1.0
        grad_Qw = np.outer(cur_phi, act_vec)

        self.Z = (self.lamb * self.gamma) * self.Z + grad_Qw

        # ==
        # Update Q function
        cur_q = self.compute_Q_value(cur_phi, cur_act)
        nex_q = self.compute_Q_value(nex_phi, nex_act)

        td_err = rew + (self.gamma * nex_q) - cur_q

        del_Wq = td_err * self.Z
        self.Wq = self.Wq + (self.lr * del_Wq)

        # ==
        # TODO log losses?
        self.log_dict['value_errors'].append(td_err)
        pass

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
    agent = LambdaAgent(n_states=5)

    print(agent)
    print(agent.V)
