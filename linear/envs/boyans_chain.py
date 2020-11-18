# ============================================================================
# Boyan's chain experiment, from:
# https://papers.nips.cc/paper/3092-ilstd-eligibility-traces-and-convergence-analysis.pdf
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class BoyansChainEnv(gym.Env):
    """
    Description:

    State space:

    Action:
    """

    def __init__(self, seed=0):
        self.n_states = 13
        self.feature_dim = 4

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)
        self.state = self.n_states

    def step(self, action):
        """
        NOTE: everything below is hard-coded specifically for the
              13-state chain with 4-dim features
        :return:
        """
        # ==
        # Update state
        reward = -3.0
        done = False

        if self.state > 2:
            cur_trans = self.rng.choice([-1, -2])
            self.state = self.state + cur_trans
        elif self.state == 2:
            self.state = 1
            reward = -2.0
        elif self.state == 1:
            self.state = 0
            reward = 0.0
            done = True
        else:
            done = True  # stay on the zero-th state

        phi = self.state_2_features(self.state)

        return phi, reward, done, {}

    def state_2_features(self, state):
        """
        Calculate the feature vector from a given state number
        :param state: integer index
        :return: feature vector
        """
        # ==
        # Generate state indeces
        s_i = (13 - state) // 4
        s_j = (13 - state) % 4

        # ==
        # Generate features
        phi = np.zeros((self.feature_dim,),
                       dtype=np.float)

        if s_i < 3:
            phi[s_i] = 1 - (0.25 * s_j)
            phi[s_i + 1] = (0.25 * s_j)
        else:
            if s_j == 0:
                phi[s_i] = 1.0

        return phi

    def reset(self):
        self.state = self.n_states
        phi = np.zeros((self.feature_dim,),
                       dtype=np.float)
        phi[0] = 1.0

        return phi

    def get_num_states(self):
        """
        Helper function to get the number of underlying states in this env
        NOTE: +1 to include the termination state
        :return: int of number of states
        """
        return self.n_states + 1

    def get_transition_matrix(self):
        """
        Helper function to return the transition matrix for this env
        :return: (self.n_states+1, self.n_states+1) np matrix
        """
        # Transition matrix
        p_n_states = self.n_states + 1
        P_trans = np.zeros((p_n_states, p_n_states))
        for i in reversed(range(3, p_n_states)):
            P_trans[i, i - 1] = 0.5
            P_trans[i, i - 2] = 0.5
        P_trans[2, 1] = 1.0
        P_trans[1, 0] = 1.0

        return P_trans

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE: charactering each state's reward by the expected reward
              from transitioning out of the state. This may be slightly
              different from the TD way of solving for state values.
        :return: (self.n_states+1,) vector
        """
        p_n_states = self.n_states + 1
        R_fn = np.ones(p_n_states) * (-3.0)
        R_fn[2] = -2.0
        R_fn[1] = 0.0
        R_fn[0] = 0.0

        return R_fn

    def solve_linear_reward_parameters(self):
        """
        Helper function to solve for the best-fit linear parameters for the
        reward function.
        :return: (self.feature_dim, ) parameters for reward fn
        """
        p_n_states = self.n_states + 1

        # Get feature matrix (p_n_states, feature_dim)
        phi_mat = np.empty((p_n_states, self.feature_dim))
        for s_n in range(p_n_states):
            phi_mat[s_n] = self.state_2_features(s_n)

        # Get reward function
        r_vec = self.get_reward_function()

        # Solve parameters from Moore–Penrose inverse
        phi_mat_T = np.transpose(phi_mat)
        mp_inv = np.linalg.inv((phi_mat_T @ phi_mat)) @ phi_mat_T
        bf_Wr = mp_inv @ r_vec

        return bf_Wr

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    # FOR TESTING ONLY
    print('hello')

    # seed = np.random.randint(100)
    # print('numpy seed:', seed)
    # np.random.seed(seed)

    env = BoyansChainEnv()

    print('=== set-up ===')
    print(env)
    print(env.action_space)
    print(env.observation_space)

    print('=== start ===')
    cur_obs = env.reset()
    print(env.state, cur_obs)

    for step in range(13):
        action = env.action_space.sample()
        cur_obs, reward, done, info = env.step(action)

        print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, {info}')
