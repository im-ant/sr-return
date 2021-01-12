# ============================================================================
# Perfect Binary Tree Environment to evaluate a "Fan-out" / "Broadcasting"
# environment's effect on credit assignment 
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class BipolarChainEnv(gym.Env):
    """
    Description: Bipolar Chain Environment with rich unsmooth rewards

    State space: tabular for now

    Action:
    """

    def __init__(self, n_states=20, reward_magnitude=10, seed=0):
        """
        TODO write docs
        """

        # Attributes
        self.n_states = n_states
        self.reward_magnitude = reward_magnitude
        self.feature_dim = self.n_states  # tabular, maybe use non-tabular / approximate?

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)  # dummy
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.feature_dim),
            high=np.ones(self.feature_dim),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)
        self.state = 1  # first state

    def step(self, action):
        """
        Transition in the random binary tree
        :return:
        """

        # ==
        # Transition, reward and termination
        done = False
        reward = self.get_current_reward(self.state)

        # ==
        # Transition
        if self.state < self.n_states:
            # In chain
            self.state += 1
        elif self.state == self.n_states:
            # Pre-terminal
            done = True
            self.state += 1
        else:
            # Termination
            done = True

        # ==
        # Features
        phi = self.state_2_features(self.state)

        return phi, reward, done, {}

    def get_current_reward(self, state):
        """
        Method to get the reward of exiting a state
        :return: float
        """
        # ==
        # Post-termiantion
        if self.state > self.n_states:
            return 0

        # ==
        # Pre-termination
        is_odd = ((state % 2) == 1)
        if is_odd:
            return 1.0 * self.reward_magnitude
        else:
            return -1.0 * self.reward_magnitude

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
        for s_idx in range(self.n_states-1):
            P_trans[s_idx, s_idx+1] = 1.0

        return P_trans

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE this is very hard-coded.
        :return: (self.n_states) vector
        """
        R_fn = np.zeros(self.n_states)

        for s_idx in range(self.n_states):
            s_num = s_idx + 1
            s_rew = self.get_current_reward(s_num)
            R_fn[s_idx] = s_rew

        return R_fn

    def get_feature_matrix(self):
        """
        Helper function to get the state to features mapping matrix
        :return: (self.n_states, self.feature_dim) np matrix
        """
        phi_mat = np.identity(self.n_states)

        return phi_mat

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


if __name__ == '__main__':
    seed = np.random.randint(100)
    print('numpy seed:', seed)
    # np.random.seed(seed)

    # add back stuff about env and running env?
    env = BipolarChainEnv(n_states=10,
                          reward_magnitude=10,
                          seed=seed)
    np.set_printoptions(precision=3, suppress=True)

    P = env.get_transition_matrix()
    print('trans mat shape', np.shape(P))
    print(P)

    R = env.get_reward_function()
    print('rew function shape', np.shape(R))
    print(R)

    # Solve for tabular value fn
    n_states = env.get_num_states()
    gamma = 0.999
    c_mat = (np.identity(n_states) - (gamma * P))
    sr_mat = np.linalg.inv(c_mat)
    v_fn_tab = sr_mat @ R

    print('tabular SR', np.shape(sr_mat))
    print(sr_mat)
    print('tabular value function:')
    print(v_fn_tab)

    # ==
    print('=== Matrices ===')
    print(env.get_transition_matrix()[0:8, 0:8])
    print(env.get_reward_function())
    print(env.get_feature_matrix()[0:8, 0:8])

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

        for step in range(7):
            action = env.action_space.sample()
            cur_obs, reward, done, info = env.step(action)

            print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, {info}')












