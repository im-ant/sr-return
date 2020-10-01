# =============================================================================
# Offline expected successor trace algorithm
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np

from algos.base import BaseAgent
from algos.lambda_agent import LambdaAgent


class STraceAgent(LambdaAgent):
    def __init__(self, n_states,
                 gamma=0.9,
                 lamb=0.8,
                 lr=0.1,
                 seed=0):
        # TODO define more arguments
        """
        TODO define arguments
        """
        super().__init__(n_states, gamma=gamma, lamb=lamb, lr=lr, seed=0)

        # Initialize successor trace matrix
        # TODO: should I initialize diagonals to be one? how to initialize this?
        #       i.e. can we assume there is always self occupancy so 1 is a good init?
        #       or maybe even 1 everywhere?
        self.sTrace = np.ones((self.n_states, self.n_states))
        # np.fill_diagonal(self.sTrace, 1)

        # TODO: Successor trace should have its separate learning rate

    def _optimize_model(self) -> None:
        # ==
        # Learning

        # Unpack trajectory and estimate values
        r_traj = self.traj['r']
        s_traj = self.traj['s']

        # Learn expected forward ("successor") trace
        self._update_successor_trace(s_traj)

        # ==
        # Compute the one-step TD errors
        td_errs = np.zeros(len(s_traj))
        for t in range(len(s_traj)):
            s_t = s_traj[t]

            # Get the next step value estimate
            if (t + 1) < len(s_traj):
                v_tp1 = self.V[s_traj[t + 1]]
            else:
                v_tp1 = 0.0

            # one-step TD error at s_k
            td_err = (r_traj[t] + (self.gamma * v_tp1)
                      - self.V[s_t])
            td_errs[t] = td_err

        # TODO PLAN
        # - turn the above into a fixed length vector via averaging over TD error at each state?
        # - then update (matrix form) each state via the TD error

        # ==
        # Update value function based on the successor trace

        # Iterate pairs t = {0, ..., T-1}
        for t in range(len(s_traj)):
            v_t_delta = 0.0
            s_t = s_traj[t]

            # Accumulate TD errors from k = {t, ..., T-1}
            # TODO update algorithm here. Need to sample differently for stability
            # TODO replace with buffer and density estimate in future?
            for k in range(t, len(s_traj)):
                s_k = s_traj[k]
                v_t_delta += self.sTrace[s_t, s_k] * td_errs[k]

            v_t_delta = v_t_delta / (len(s_traj)-t)

            # Update value function
            # NOTE: not entirely sure if this is offline
            self.V[s_t] += self.lr * v_t_delta

        # ==
        # TODO log losses?
        pass

    def _update_successor_trace(self, s_traj):
        """
        :param s_traj:
        :return:
        """
        # TODO: should be able to learn this using lambda return but for now use TD(0)
        # TODO: can even sample from the state space?
        # TODO: make below into linear rather than quadratic time algorithm?

        # Iterate pairs of states with indeces t = {0, ..., T-2}
        # and k = {t, ..., T-1}
        for t in range(len(s_traj)-1):
            for k in range(t, len(s_traj)):
                s_t, s_tp1, s_k = s_traj[t], s_traj[t+1], s_traj[k]

                mDelta = (float(s_t == s_k)
                          + ((self.lamb*self.gamma) * self.sTrace[s_tp1, s_k])
                          - self.sTrace[s_t, s_k])

                # NOTE: using the same learning rate
                self.sTrace[s_t, s_k] += self.lr * mDelta

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = STraceAgent(n_states=5)

    print(agent)
    print(agent.V)
