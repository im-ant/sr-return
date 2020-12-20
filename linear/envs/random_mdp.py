# ============================================================================
# Random MDP
#
# Transition probabilities from each state are sampled as independent
# dirichlet distributions with the same concentration parameter. Rewards are
# sampled independent from a normal Gaussian.
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class RandomMDPEnv(gym.Env):
    """
    Description: MDP with randomly sampled transitions and rewards

    State space:

    Action:
    """

    def __init__(self, n_states=13, seed=0):
        self.n_states = n_states
        self.feature_dim = n_states

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]*self.feature_dim),
            high=np.array([1.0]*self.feature_dim),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)

        # ==
        # Construct MDP

        # Sample transition matrix
        dir_concen = 0.5  # concentration parameter  TODO make into hyperparam?
        self.pMat = self.rng.dirichlet(
            alpha=([dir_concen] * self.n_states),
            size=self.n_states
        )  # (n_states, n_states)

        # Sample reward function
        # TODO: should it be sparse or rich here?
        self.rVec = self.rng.standard_normal(size=self.n_states)

        # ==
        # Sample initial state uniformaly
        self.state = self.rng.choice(self.n_states)

    def step(self, action):
        """
        :return:
        """

        # TODO: make into episodic setting? or keep as continuing?
        done = False

        # Get state exit reward
        reward = self.rVec[self.state]

        # Update transition
        new_state_idx = self.rng.choice(self.n_states,
                                        p=self.pMat[self.state])
        self.state = new_state_idx

        # Get feature
        phi = self.state_2_features(self.state)

        return phi, reward, done, {}

    def state_2_features(self, state):
        """
        For now, one-hot
        :param state: integer index
        :return: feature vector
        """
        phi = np.zeros(self.n_states)
        phi[state] = 1.0

        # TODO: make not one-hot?

        return phi

    def reset(self):
        self.state = self.rng.choice(self.n_states)
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
        return self.pMat

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE: charactering each state's reward by the expected reward
              from transitioning out of the state.
        :return: (self.n_states,) vector
        """
        return self.rVec

    def solve_linear_reward_parameters(self):
        """
        NOTE DEC 20 not sure if valid. Recheck.

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
    np.random.seed(seed)

    env = RandomMDPEnv(n_states=5, seed=seed)

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
