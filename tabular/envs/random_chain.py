# ============================================================================
# Random walk chain expeirment, from Example 6.2 in Sutton & Barto 2018
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np

class RandomChainEnv(gym.Env):
    """
    Description:
        The agent starts in a centre of a chain and randomly transition to
        the left or right.

    State space:
        Tabular: {0, ..., n-1}

    Action: 0
    """

    def __init__(self, n_states=5, seed=0):
        #
        # ==
        # Attributes
        assert n_states % 2 == 1
        self.n_states = n_states


        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)
        self.observation_space = gym.spaces.Discrete(n=self.n_states)

        # ==
        # Initialize state in the middle
        self.state = int(self.n_states // 2)

        # RNG
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        """
        One step transition
        :param action:
        :return:
        """
        # Random transition if within good range
        if 0 <= self.state < self.n_states:
            rand_trans = self.rng.choice([-1, 1])
            self.state = self.state + rand_trans

        #
        reward = 0.0
        done = False

        if self.state < 0:
            obs = 0
            done = True
        elif self.state >= self.n_states:
            obs = self.n_states - 1
            reward = 1.0
            done = True
        else:
            obs = self.state

        return obs, reward, done, {}

    def reset(self):
        """
        :return: initial observation
        """
        # Reset state
        self.state = int(self.n_states // 2)
        obs = self.state

        return obs

    def render(self):
        """
        :return:
        """
        pass

    def close(self):
        pass


if __name__ == '__main__':
    # FOR TESTING ONLY
    print('hello')

    #seed = np.random.randint(100)
    #print('numpy seed:', seed)
    #np.random.seed(seed)

    env = RandomChainEnv(n_states=5)

    print('=== set-up ===')
    print(env)
    print(env.action_space)
    print(env.observation_space)


    print('=== start ===')
    cur_obs = env.reset()
    print(env.state, cur_obs)

    for step in range(10):
        action = env.action_space.sample()
        cur_obs, reward, done, info = env.step(action)

        print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, info')
