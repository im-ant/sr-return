# ============================================================================
# Random walk chain experiment, adopted from Sutton & Barto Book
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class RandomWalkChainEnv(gym.Env):
    """
    Description:

    State space:

    Action:
    """

    def __init__(self, seed=0):
        self.n_states = 20  # NOTE fix for now and set to 20 for 19-state chain
        self.feature_dim = self.n_states-1  # NOTE: using tabular representation

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.feature_dim),
            high=np.ones(self.feature_dim),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)
        self.state = int(self.n_states / 2)  # start in middle

    def step(self, action):
        """
        Transition in the random walk chain
        :return:
        """
        # ==
        # Update state
        reward = 0.0
        done = False

        # Transition
        if 1 <= self.state < self.n_states:
            cur_trans = self.rng.choice([-1, +1])
            self.state = self.state + cur_trans

        # Features, rewards and termination
        phi = self.state_2_features(self.state)
        if self.state == self.n_states:
            reward = 1.0
        if not (1 <= self.state < self.n_states):
            done = True

        return phi, reward, done, {}

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
        self.state = int(self.n_states / 2)  # start in middle

        phi = self.state_2_features(self.state)
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
        for i in range(1, (p_n_states-1)):
            P_trans[i, i + 1] = 0.5
            P_trans[i, i - 1] = 0.5
        P_trans[0, 0] = 0.0
        P_trans[self.n_states, self.n_states] = 0.0

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
        R_fn = np.zeros(p_n_states)
        R_fn[-1] = 1.0

        return R_fn

    def get_feature_matrix(self):
        """
        Helper function to get the state to features mapping matrix
        :return: (self.n_states+1, self.feature_dim) np matrix
        """
        # Get feature matrix (p_n_states, feature_dim)
        p_n_states = self.n_states + 1
        phi_mat = np.empty((p_n_states, self.feature_dim))
        for s_n in range(p_n_states):
            phi_mat[s_n] = self.state_2_features(s_n)
        return phi_mat

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

    seed = np.random.randint(100)
    print('numpy seed:', seed)
    # np.random.seed(seed)

    # TODO add back stuff about env and running env?
    env = RandomWalkChainEnv(seed=seed)

    P = env.get_transition_matrix()
    print('trans mat shape', np.shape(P))
    # print(P)

    R = env.get_reward_function()
    print('rew function shape', np.shape(R))
    # print(R)

    # Solve for tabular value fn
    n_states = env.get_num_states()
    gamma = 1.0
    c_mat = (np.identity(n_states) - (gamma * P))
    sr_mat = np.linalg.inv(c_mat)
    v_fn_tab = sr_mat @ R

    np.set_printoptions(precision=3)

    print('tabular SR', np.shape(sr_mat))
    # print(sr_mat)
    print('tabular value function:')
    print(v_fn_tab)

    # ==
    # Run a few
    print('=== set-up ===')
    print('env', env)
    print('action space', env.action_space)
    print('obs space', env.observation_space)

    print('=== start ===')
    cur_obs = env.reset()
    print(f's: {env.state}, obs: {cur_obs}')

    for step in range(30):
        action = env.action_space.sample()
        cur_obs, reward, done, info = env.step(action)

        print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, {info}')












