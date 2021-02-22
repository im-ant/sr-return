# =============================================================================
# Q Learning agent
#
# Author: Anthony G. Chen
# =============================================================================

from collections import namedtuple
from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseLinearAgent
from utils.optim import RMSProp

LogTupStruct = namedtuple(
    'LogTupStruct',
    field_names=['lamb', 'lr', 'policy_epsilon', 'optim_kwargs',
                 'value_loss_avg']
)


class QAgent(BaseLinearAgent):
    def __init__(self, feature_dim,
                 num_actions,
                 gamma=0.9,
                 lamb=0.0,
                 lr=0.1,
                 policy_epsilon=0.3,
                 optim_kwargs=None,
                 seed=0):

        """
        TODO define arguments
        """
        super().__init__(feature_dim, num_actions, gamma=gamma, lr=lr, seed=seed)
        self.lamb = lamb  # TODO implement?

        self.policy_epsilon = policy_epsilon

        # ==
        # Initialize Q function parameter and optimizer
        self.Wq = self.rng.uniform(
            low=0.0, high=1e-5,
            size=(self.feature_dim, self.num_actions)
        )
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        self.Wq_optim = RMSProp(self.Wq, lr=lr, **optim_kwargs)

        # ==
        # Initialize trace (TODO implement and check validity?)
        self.Z = np.zeros((self.feature_dim, self.num_actions))

        # ==
        # For logging
        self.logTupStruct = LogTupStruct

    def begin_episode(self, phi):
        action = super().begin_episode(phi)

        self.log_dict = {
            'value_errors': [],
        }

        # Reset eligibility trace  TODO implement?
        self.Z *= 0.0

        return action

    def step(self, phi_t: np.array, reward: float, done: bool) -> int:
        """
        Take step in the environment
        :param phi_t: vector of feature observation
        :param reward: scalar reward
        :param done: boolean for whether episode is finished
        :return: integer action index
        """

        # New action and storage
        new_act = self._select_action(phi_t)
        if not done:
            self.traj['phi'].append(phi_t)
            self.traj['a'].append(new_act)
        self.traj['r'].append(reward)

        # ==
        # Learning

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
        # Update trace
        act_vec = np.zeros(self.num_actions)
        act_vec[cur_act] = 1.0
        grad_Qw = np.outer(cur_phi, act_vec)

        self.Z = (self.lamb * self.gamma) * self.Z + grad_Qw

        # ==
        # Update Q function
        if not done:
            nex_phi = self.traj['phi'][t_idx + 1]
            nex_q = np.max(nex_phi @ self.Wq)
        else:
            nex_q = 0.0

        cur_q = self.compute_Q_value(cur_phi, cur_act)
        td_err = rew + (self.gamma * nex_q) - cur_q

        # Parameter updates
        del_Wq = td_err * self.Z
        ada_del_Wq = self.Wq_optim.step(del_Wq)
        self.Wq = self.Wq + ada_del_Wq

        # ==
        # Logging losses
        self.log_dict['value_errors'].append(td_err)

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

        # ==
        # Simple epsilon-greedy policy

        rand_eps = self.policy_epsilon

        # Random action with prob epsilon
        if self.rng.binomial(n=1, p=rand_eps) == 1:
            action = self.rng.choice(self.num_actions)
        # Greedy action
        else:
            q_vec = phi @ self.Wq  # (n_actions, )
            action = np.argmax(q_vec)

        return action

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = LambdaAgent(n_states=5)

    print(agent)
    print(agent.V)
