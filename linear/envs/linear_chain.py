# ============================================================================
# Simple linear chain environment to illustrate a point.
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class SimpleLinearChainEnv(gym.Env):
    """
    Description: Simple linear chain with reward at the end.
    State space: NOTE tabular for now
    Action: 1
    """

    def __init__(self,
                 n_states=7,
                 skip_prob=0.1,
                 terminal_reward_stdev=0.1,
                 seed=0):
        """
        TODO write docs
        """

        # Attributes
        self.n_states = n_states
        self.feature_dim = self.n_states  # tabular

        self.skip_prob = skip_prob
        self.terminal_reward_mean = 1.0
        self.terminal_reward_stdev = terminal_reward_stdev

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)  # dummy
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.feature_dim),
            high=np.ones(self.feature_dim),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)
        self.state = 1

    def step(self, action):
        """
        Transition in the random binary tree
        :return:
        """

        # ==
        # Transition, reward and termination
        done = False
        reward = self.get_current_reward(self.state)

        if self.state >= self.n_states:
            # End and absorbing nodes
            done = True
            if self.state == self.n_states:
                self.state += 1
        elif self.state == (self.n_states - 1):
            # Pre-terminal state
            self.state = self.n_states
        else:
            s_delta = self.rng.choice(
                [1, 2], p=[(1 - self.skip_prob), self.skip_prob]
            )
            self.state += s_delta

        # ==
        # Features
        phi = self.state_2_features(self.state)

        return phi, reward, done, {}

    def get_current_reward(self, state):
        """
        Method to get the reward of exiting a state
        :return: float
        """
        if state == self.n_states:
            rew = (self.terminal_reward_mean +
                   self.rng.normal(scale=self.terminal_reward_stdev))
            return rew
        else:
            return 0.0

    def state_2_features(self, state):
        """
        Calculate the feature vector from a given state number
        :param state: integer index
        :return: feature vector
        """
        # Simple tabular features
        phi = np.zeros((self.feature_dim,),
                       dtype=np.float)

        # Set index
        s_idx = state - 1
        if 0 <= s_idx < self.feature_dim:
            phi[s_idx] = 1.0

        return phi

    def reset(self):
        self.state = 1
        phi = self.state_2_features(self.state)
        return phi

    def get_num_states(self):
        """
        Helper function to get the number of underlying states in this env
        NOTE: +1 to include the termination state
        :return: int of number of states
        """
        return self.n_states

    def get_transition_matrix(self):
        """
        Helper function to return the transition matrix for this env
        :return: (self.n_states, self.n_states) np matrix
        """
        # Transition matrix
        P_trans = np.zeros((self.n_states, self.n_states))

        # Fill
        for i in range(self.n_states):
            s_num = i + 1
            if s_num >= self.n_states:
                pass
            elif s_num == (self.n_states - 1):
                P_trans[i, i+1] = 1.0
            else:
                P_trans[i, i+1] = 1 - self.skip_prob
                P_trans[i, i+2] = self.skip_prob

        return P_trans

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE this is very hard-coded.
        :return: (self.n_states) vector
        """
        R_fn = np.zeros(self.n_states)
        R_fn[-1] = self.terminal_reward_mean

        return R_fn

    def get_feature_matrix(self):
        """
        Helper function to get the state to features mapping matrix
        :return: (self.n_states, self.feature_dim) np matrix
        """
        phi_mat = np.empty((self.n_states, self.feature_dim))
        for s_idx in range(self.n_states):
            cur_phi = self.state_2_features((s_idx + 1))
            phi_mat[s_idx, :] = cur_phi

        return phi_mat

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    seed = np.random.randint(100)
    print('numpy seed:', seed)
    # np.random.seed(seed)

    # add back stuff about env and running env?
    env = SimpleLinearChainEnv(n_states=13,
                               seed=seed)

    P = env.get_transition_matrix()
    print('trans mat shape', np.shape(P))
    print(P)

    R = env.get_reward_function()
    print('rew function shape', np.shape(R))
    print(R)

    # Solve for tabular value fn
    n_states = env.get_num_states()
    gamma = 0.9
    c_mat = (np.identity(n_states) - (gamma * P))
    sr_mat = np.linalg.inv(c_mat)
    v_fn_tab = sr_mat @ R

    np.set_printoptions(precision=3)

    print('tabular SR', np.shape(sr_mat))
    print(sr_mat)
    print('tabular value function:')
    print(v_fn_tab)

    # ==
    # print('=== Matrices ===')
    # print(env.get_transition_matrix()[0:8, 0:8])
    # print(env.get_reward_function())
    # print(env.get_feature_matrix()[0:8, 0:8])

    # ==
    # Run a few
    print('=== set-up ===')
    print('env', env)
    print('action space', env.action_space)
    print('obs space', env.observation_space)

    print('=== start ===')

    for epis_idx in range(3):
        cur_obs = env.reset()
        print(f'[EPIS]: {epis_idx}, s: {env.state}, obs: {cur_obs}')

        for step in range(8):
            action = env.action_space.sample()
            cur_obs, reward, done, info = env.step(action)

            print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, {info}')
