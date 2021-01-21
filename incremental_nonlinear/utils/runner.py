# ============================================================================
# Runner class
#
# Inspired by a combination of runner and sampler classes from rlpyt,
# simplified for the incremental online setting.
#
# Author: Anthony G Chen
# ============================================================================
from collections import namedtuple

import torch

TransitionTuple = namedtuple('transition', 'state, last_state, action, reward, is_terminal')


class IncrementalOnlineRunner(object):
    def __init__(self, algo, EnvCls, env_kwargs, device='cpu'):

        self.algo = algo
        self.EnvCls = EnvCls
        self.env_kwargs = env_kwargs
        self.n_steps = 1e4
        self.device = device

        print('Initializing environment...')
        self.environment = self.EnvCls(**self.env_kwargs)
        print(self.environment)

        print('Initializing algo...')
        self.algo.initialize(self.environment, self.device)
        print(self.algo)

    def get_state(self, s):
        """
        Helper method to permute the state axis (from MinAtar)
        :param s:
        :return:
        """
        return (torch.tensor(s, device=self.device).permute(2, 0, 1)).unsqueeze(0).float()

    def world_dynamics(self, s, env, network):
        # network(s)[0] specifies the policy network, which we use to draw an action according to a multinomial
        # distribution over axis 1, (axis 0 iterates over samples, and is unused in this case. torch._no_grad()
        # avoids tracking history in autograd.
        with torch.no_grad():
            action = torch.multinomial(network(s)[0], 1)[0]  # TODO change to model

        # Act according to the action and observe the transition and reward
        reward, terminated = env.act(action)

        # Obtain s_prime
        s_prime = self.get_state(env.state())

        return s_prime, action, torch.tensor([[reward]], device=self.device).float(), torch.tensor([[terminated]],
                                                                                                   device=self.device)

    def train(self, n_steps):
        """
        Main training loop
        :return:
        """
        episode_count = 0
        steps = 0
        exp_average_return = 0.0

        env = self.environment
        network = self.algo.model

        while steps < n_steps:
            # ==
            # Initialize environment and start state
            current_episode_return = 0.0

            env.reset()
            s = self.get_state(env.state())

            is_terminated = False
            s_last = None
            r_last = None
            term_last = None

            while (not is_terminated) and steps < n_steps:
                # Generate data
                s_prime, action, reward, is_terminated = self.world_dynamics(s, env, network)
                sample = TransitionTuple(s, s_last, action, r_last, term_last)

                self.algo.optimize_agent(sample, steps)

                current_episode_return += reward.item()
                steps += 1

                # Continue the process
                s_last = s
                r_last = reward
                term_last = is_terminated
                s = s_prime

            # Increment the episodes
            episode_count += 1
            sample = TransitionTuple(s, s_last, action, r_last, term_last)

            self.algo.optimize_agent(sample, steps)
            self.algo.clear_eligibility_traces()  # clear trace

            # Save the return for each episode  # TODO not sure what to do here
            #returns.append(G)  # TODO necessary?
            #frame_stamps.append(t)  # TODO necessary?

            # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
            exp_average_return = ((0.99 * exp_average_return)
                                  + (0.01 * current_episode_return))

            # TODO: add logging and saving
            print(f'Total steps: {steps}; '
                  f'Episode: {episode_count}; '
                  f'Return: {current_episode_return}; '
                  f'Exp Avg Return: {exp_average_return}')
