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
from algos.lambda_agent import LambdaAgent


class STraceAgent(LambdaAgent):
    def __init__(self, n_states,
                 gamma=0.9,
                 lamb=0.8,
                 lr=0.1,
                 s_prop_sample=0.10,
                 use_true_s_mat=False,
                 use_rand_s_mat=False,
                 use_true_r_fn=False,  # dummy var
                 seed=0):
        # TODO define more arguments
        """
        TODO define arguments
        """
        super().__init__(n_states, gamma=gamma, lamb=lamb, lr=lr, seed=seed)
        self.use_true_s_mat = use_true_s_mat
        self.use_rand_s_mat = use_rand_s_mat
        self.s_prop_sample = s_prop_sample  # proportion of traj to sample for S-mat learning

        # ==
        # Initialize successor trace matrix
        avg_p = 0.5  # i.e. for all states there is a .5 chance of visiting each other
        s_r = min((self.gamma * self.lamb), 0.95)  # bound this for numerical stability
        s_init = (1.0 / (1.0 - s_r)) * avg_p  # smart init?

        self.S_mat = np.ones((self.n_states, self.n_states)) * s_init
        # self.S_mat = np.zeros((self.n_states, self.n_states))

        # ==
        # Use the true successor trace optionally
        if self.use_true_s_mat:
            self.S_mat = self.solve_sr_matrix(
                num_states=self.n_states,
            )
        elif self.use_rand_s_mat:
            self.use_true_s_mat = True
            self.S_mat = np.random.rand(n_states, n_states)  # [0, 1)

    def _optimize_model(self) -> None:
        # ==
        # Learning

        # Unpack trajectory and estimate values
        r_traj = self.traj['r']
        s_traj = self.traj['s']

        # Learn expected forward ("successor") trace
        if not self.use_true_s_mat:
            self._update_successor_trace_anytime(s_traj)

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
                v_t_delta += self.S_mat[s_t, s_k] * td_errs[k]

            v_t_delta = v_t_delta / (len(s_traj) - t)

            # Update value function
            # NOTE: not entirely sure if this is offline
            self.V[s_t] += self.lr * v_t_delta

        # ==
        # TODO log losses?
        pass

    def _update_successor_trace_anytime(self, s_traj):
        """
        Anytime algorithm for updating the SR via sampling the online
        trajectory. Note this can be vastly improved with a buffer or
        simply iterating the matrix, but for fairness (and extension to
        sucessor feature we keep it this way)

        :param s_traj:
        :return:
        """
        s_lr = self.lr

        for t in range(len(s_traj)):
            if (t + 1) < len(s_traj):
                s_t, s_tp1 = s_traj[t], s_traj[t + 1]
            else:
                s_t, s_tp1 = s_traj[t], None

            # Sample traj with replacement as a heuristic
            num_sample = int(self.s_prop_sample * len(s_traj))
            s_k_vec = self.rng.choice(s_traj, size=num_sample)
            for s_k in s_k_vec:
                if s_tp1 is None:
                    mDelta = (float(s_t == s_k)
                              - self.S_mat[s_t, s_k])  # TODO: termination case?
                else:
                    mDelta = (float(s_t == s_k)
                              + ((self.lamb * self.gamma) * self.S_mat[s_tp1, s_k])
                              - self.S_mat[s_t, s_k])

                self.S_mat[s_t, s_k] += s_lr * mDelta

    def _update_successor_trace_nested(self, s_traj):
        """
        Old methods, reference only. Updates SR using a nested loop over the entire
        trajectory
        :param s_traj:
        :return:
        """
        # TODO: should be able to learn this using lambda return but for now use TD(0)
        # TODO: can even sample from the state space?
        # TODO: make below into linear rather than quadratic time algorithm?

        s_lr = self.lr  # NOTE: making a s-trace specific learning rate?
        # s_trace = copy.deepcopy(self.sTrace)  # if totally offine

        # Iterate pairs of states with indeces t = {0, ..., T-2}
        # and k = {t, ..., T-1}
        for t in range(len(s_traj) - 1):
            for k in range(t, len(s_traj)):
                s_t, s_tp1, s_k = s_traj[t], s_traj[t + 1], s_traj[k]

                mDelta = (float(s_t == s_k)
                          + ((self.lamb * self.gamma) * self.sTrace[s_tp1, s_k])
                          - self.sTrace[s_t, s_k])

                self.sTrace[s_t, s_k] += s_lr * mDelta
                # s_trace[s_t, s_k] += s_lr * mDelta  # if totally offline

        # self.sTrace = s_trace  # if totally offline

    def solve_sr_matrix(self, num_states):
        """
        Solve the SR matrix exactly for the random walk chain task
        :param num_states: number of states in the chain
        :param discount_factor: discount factor to apply
        :return:
        """
        # TODO fix bug here

        # ==
        # Construct the transition matrix
        P_mat = np.zeros((num_states, num_states))
        for i in range(num_states - 1):
            P_mat[i, i + 1] = 0.5
            P_mat[i + 1, i] = 0.5
        P_mat[0, 0] = 0.0  # 0.0 or 0.5?
        P_mat[-1, -1] = 0.0

        # ==
        # Safeguard against singular matrix
        df = self.gamma * self.lamb
        # df = min(df, 0.99999)

        # ==
        # Solve the discounted occupancy problem
        c_mat = np.identity(num_states) - (df * P_mat)
        sr_mat = np.linalg.inv(c_mat)

        return sr_mat

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = STraceAgent(n_states=5)

    print(agent)
    print(agent.V)
