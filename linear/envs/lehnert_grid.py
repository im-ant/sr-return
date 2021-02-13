# ============================================================================
# Fan-in tree to evaluate credit assignment.
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class LehnertGridWorldEnv(gym.Env):
    """
    Description: Grid World control environment as described in
                 (Lehnert et al 2017) https://arxiv.org/abs/1708.00102

    State space: 0 - up; 1 - down; 2 - left; 3 - right
    Action:
    """

    def __init__(self,
                 width=5,
                 slip_prob=0.05,
                 episode_max_length=200,
                 goal_switch_freq=None,
                 seed=0):
        """
        TODO write docs
        """

        # Attributes
        self.width = width
        self.slip_prob = slip_prob
        self.episode_max_length = episode_max_length

        self.n_states = width ** 2
        self.feature_dim = self.n_states  # tabular

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=4)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.feature_dim),
            high=np.ones(self.feature_dim),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)

        # ==
        # Goal
        self.goals = [
            (0, self.width - 1),  # top right
            (0, self.width - 2),
            (0, self.width - 3),
            (0, self.width - 4),
            (0, self.width - 5),
            (0, self.width - 4),
            (0, self.width - 3),
            (0, self.width - 2),
        ]
        self.goal_switch_freq = goal_switch_freq

        # ==
        # States
        self.state = (self.width - 1, self.width - 1)  # bottom right
        self.goal_idx = 0  # current goal index
        self.total_episode_count = 0  # count total num episodes resets
        self.current_episode_steps = 0

    def tup2idx(self, tup):
        return tup[0] * self.width + tup[1]

    def idx2tup(self, idx):
        return idx // self.width, idx % self.width

    def reset(self):
        # Set goal
        self.total_episode_count += 1
        if self.goal_switch_freq is not None:
            if self.total_episode_count % self.goal_switch_freq == 0:
                num_goals = len(self.goals)
                self.goal_idx = (self.goal_idx + 1) % num_goals

        # Set current episode
        self.state = (self.width - 1, self.width - 1)  # bottom right
        self.current_episode_steps = 0
        phi = self.state_2_features(self.state)
        return phi

    def step(self, action):
        """
        Transition in the random binary tree
        :return:
        """

        current_goal = self.goals[self.goal_idx]

        # ==
        # Transition, reward and termination
        done = False
        reward = 0.0

        # Leaf and absorbing nodes
        if self.state == current_goal:
            done = True
            reward = 1.0
        else:
            direction = None
            if self.rng.binomial(n=1, p=self.slip_prob) == 1:
                # Slip
                direction = self.rng.choice([2, 3])
            else:
                # No slip
                direction = action

            # ==
            # Move along direction
            dir2delta = {
                0: (-1, 0),  # up
                1: (+1, 0),  # down
                2: (0, -1),  # left
                3: (0, +1),  # right
            }
            row, col = self.state
            drow, dcol = dir2delta[direction]
            self.state = (
                np.clip(row + drow, 0, self.width - 1),
                np.clip(col + dcol, 0, self.width - 1),
            )

        # Additionally: terminate of time limit exceeded
        if self.current_episode_steps >= self.episode_max_length:
            done = True

        # ==
        # Features
        phi = self.state_2_features(self.state)

        self.current_episode_steps += 1

        return phi, reward, done, {}

    def get_current_reward(self, state):
        """
        Method to get the reward of exiting a state
        :return: float
        """
        raise NotImplementedError

    def state_2_features(self, state_tup):
        """
        Calculate the feature vector from a given state number
        :param state: integer index
        :return: feature vector
        """
        #
        s_idx = self.tup2idx(state_tup)

        # Tabular features
        phi = np.zeros((self.feature_dim,),
                       dtype=np.float)
        phi[s_idx] = 1.0
        # TODO: set last state to have zero feature?
        return phi

    def get_num_states(self):
        """
        Helper function to get the number of underlying states in this env
        NOTE: +1 to include the termination state
        :return: int of number of states
        """
        return self.n_states

    def get_transition_matrix(self):
        raise NotImplementedError

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE this is very hard-coded.
        :return: (self.n_states) vector
        """

        # TODO what to do with this
        R_fn = np.zeros(self.n_states)

        R_fn[0] = 1.0

        return R_fn

    def get_feature_matrix(self):
        """
        Helper function to get the state to features mapping matrix
        :return: (self.n_states, self.feature_dim) np matrix
        """

        """
        # TODO what to do with this
        phi_mat = np.empty((self.n_states, self.feature_dim))
        for s_idx in range(self.n_states):
            cur_phi = self.state_2_features((s_idx + 1))
            phi_mat[s_idx, :] = cur_phi
        
        return phi_mat
        """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


if __name__ == '__main__':
    seed = np.random.randint(100)
    print('numpy seed:', seed)
    # np.random.seed(seed)

    # add back stuff about env and running env?
    # TODO update below if I want to run test
    env = LehnertGridWorldEnv(seed=seed)

    # Solve for tabular value fn
    n_states = env.get_num_states()
    gamma = 0.9

    np.set_printoptions(precision=3)

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

            print(f'a: {action}, s: {env.state}, obs: [omitted], r:{reward}, d:{done}, {info}')
