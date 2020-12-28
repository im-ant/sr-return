# ============================================================================
# Perfect Binary Tree Environment to evaluate a "Fan-out" / "Broadcasting"
# environment's effect on credit assignment 
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class PerfBinaryTreeEnv(gym.Env):
    """
    Description: Perfect Binary Tree environment, reward only at the leaf
                 nodes.

    State space: NOTE tabular for now

    Action:
    """

    def __init__(self, depth=5, seed=0):
        """
        TODO write docs
        """

        # Attributes
        self.depth = depth
        self.n_states = (2**self.depth) - 1
        self.feature_dim = self.n_states  # tabular
        self.terminal_high_rew_prob = 0.2

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)  # dummy
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.feature_dim),
            high=np.ones(self.feature_dim),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)
        self.state = 1  # ancestor node

    def step(self, action):
        """
        Transition in the random binary tree
        :return:
        """

        # ==
        # Transition, reward and termination
        done = False
        reward = self.get_current_reward(self.state)

        # Leaf and absorbing nodes
        if self.state >= (2**(self.depth-1)):
            done = True
            if self.state < (2**self.depth):
                self.state = (2 ** self.depth)  # go to absorbing
        # Non-leaf nodes
        else:
            # Random transition to child node (non leaf nodes)
            child_idx = self.rng.choice([0, 1])
            self.state = (self.state * 2) + child_idx

        # ==
        # Features
        phi = self.state_2_features(self.state)

        return phi, reward, done, {}

    def get_current_reward(self, state):
        """
        Method to get the reward of exiting a state
        :return: float
        """

        # For final absorption state (should never need this if
        # the episode terminates on done)
        if state == (2**self.depth):
            return 0.0
        # For non-leaf states
        elif state < (2**(self.depth-1)):
            return 0.0  # TODO: default reward 0 or 1?

        # ==
        # For leaf states

        hig_rew = state - (2**(self.depth-1))
        low_rew = 0.0

        p = self.terminal_high_rew_prob
        rew = self.rng.choice([hig_rew, low_rew], p=[p, (1-p)])
        return rew

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
        for s_num in range(1, (2**(self.depth-1))):
            s_idx = s_num - 1
            child1_idx = s_num*2 - 1
            child2_idx = s_num*2

            P_trans[s_idx, child1_idx] = 0.5
            P_trans[s_idx, child2_idx] = 0.5

        return P_trans

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE this is very hard-coded.
        :return: (self.n_states) vector
        """
        R_fn = np.zeros(self.n_states)
        p = self.terminal_high_rew_prob

        # Iterate over leaf states
        for l_state in range((2**(self.depth-1)), (2 ** self.depth)):
            cur_hig_rew = l_state - (2**(self.depth-1))
            cur_low_rew = 0.0
            cur_avg_rew = (p * cur_hig_rew) + ((1-p) * cur_low_rew)

            sIdx = l_state - 1
            R_fn[sIdx] = cur_avg_rew

        return R_fn

    def get_feature_matrix(self):
        """
        Helper function to get the state to features mapping matrix
        :return: (self.n_states+1, self.feature_dim) np matrix
        """
        phi_mat = np.empty((self.n_states, self.feature_dim))
        for s_idx in range(self.n_states):
            cur_phi = self.state_2_features((s_idx+1))
            phi_mat[s_idx, :] = cur_phi

        return phi_mat

    def solve_linear_reward_parameters(self):
        """
        Helper function to solve for the best-fit linear parameters for the
        reward function.
        :return: (self.feature_dim, ) parameters for reward fn
        """
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    seed = np.random.randint(100)
    print('numpy seed:', seed)
    # np.random.seed(seed)

    # add back stuff about env and running env?
    env = PerfBinaryTreeEnv(depth=3,
                            seed=seed)

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

        for step in range(5):
            action = env.action_space.sample()
            cur_obs, reward, done, info = env.step(action)

            print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, {info}')












