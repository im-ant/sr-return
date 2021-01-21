# ============================================================================
# Runner class
#
# Inspired by a combination of runner and sampler classes from rlpyt,
# simplified for the incremental online setting.
#
# Author: Anthony G Chen
# ============================================================================
from collections import namedtuple
import dataclasses
import logging
import os

import torch


# ==
# Logging related
@dataclasses.dataclass
class LogTupStruct:
    episode_count: int = None
    total_steps: int = None
    episode_return: float = None
    exp_average_return: float = None
    value_loss: float = None
    reward_loss: float = None
    sf_loss: float = None
    et_loss: float = None


def init_logger(logging_path: str) -> logging.Logger:
    """
    Initializes the path to write the log to
    :param logging_path: path to write the log to
    :return: logging.Logger object
    """
    logger = logging.getLogger('Experiment')

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logging_path)
    formatter = logging.Formatter('%(asctime)s||%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def add_log_dict(sum_dict, new_dict) -> dict:
    """
    Helper method to average two dicts' logs
    :param sum_dict:
    :param new_dict:
    :return:
    """
    if sum_dict is None:
        return new_dict

    for k in new_dict:
        val = new_dict[k]
        if isinstance(val, float) or isinstance(val, int):
            # Sum numerals
            if k in sum_dict:
                sum_dict[k] += val
            else:
                sum_dict[k] = val
        else:
            # Update numerals or non-existing numerals
            sum_dict[k] = val
    return sum_dict


def avg_log_dict(sum_dict, n):
    for k in sum_dict:
        # Skip no-numerals
        val = sum_dict[k]
        if not (isinstance(val, float) or isinstance(val, int)):
            continue
        sum_dict[k] = val / n
    return sum_dict


# ==
# For training samples
TransitionTuple = namedtuple('transition', 'state, last_state, action, reward, is_terminal')


# ==
# Runner object
class IncrementalOnlineRunner(object):
    def __init__(self, algo, EnvCls, env_kwargs,
                 log_interval_episodes=10,
                 log_dir_path=None,
                 device='cpu'):

        self.algo = algo
        self.EnvCls = EnvCls
        self.env_kwargs = env_kwargs
        self.device = device

        self.log_interval_episodes = log_interval_episodes
        self.log_dir_path = log_dir_path

        print('Initializing environment...')
        self.environment = self.EnvCls(**self.env_kwargs)
        print(self.environment)

        print('Initializing algo...')
        self.algo.initialize(self.environment, self.device)
        print(self.algo)

        # ==
        # Counter items
        self.exp_average_return = 0.0
        self.log_interval_sum_return = 0.0
        self.log_interval_sum_dict = None

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
        # ==
        # Setting up logging
        dc_fields = [k for k in vars(LogTupStruct())]  # temp object to get name
        log_title_str = '|'.join(dc_fields)
        if self.log_dir_path is not None:
            log_path = os.path.join(self.log_dir_path, 'progress.csv')
            logger = init_logger(log_path)
            logger.info(log_title_str)
        else:
            print(log_title_str)
            logger = None

        # ==
        episode_count = 0
        total_steps = 0
        exp_average_return = 0.0

        env = self.environment
        network = self.algo.model

        while total_steps < n_steps:
            # ==
            # Initialize environment and start state
            current_episode_steps = 0
            current_episode_return = 0.0
            cur_sum_dict = None

            env.reset()
            s = self.get_state(env.state())

            is_terminated = False
            s_last = None
            r_last = None
            term_last = None

            while (not is_terminated) and total_steps < n_steps:
                # Generate data
                s_prime, action, reward, is_terminated = self.world_dynamics(s, env, network)
                sample = TransitionTuple(s, s_last, action, r_last, term_last)

                out_dict = self.algo.optimize_agent(sample, total_steps)

                current_episode_return += reward.item()
                current_episode_steps += 1
                total_steps += 1
                cur_sum_dict = add_log_dict(cur_sum_dict, out_dict)

                # Continue the process
                s_last = s
                r_last = reward
                term_last = is_terminated
                s = s_prime

            # Increment the episodes
            episode_count += 1
            sample = TransitionTuple(s, s_last, action, r_last, term_last)

            out_dict = self.algo.optimize_agent(sample, total_steps)
            self.algo.clear_eligibility_traces()  # clear trace

            cur_sum_dict = add_log_dict(cur_sum_dict, out_dict)
            cur_avg_dict = avg_log_dict(cur_sum_dict, current_episode_steps)

            # ==
            # Write logs
            self.write_log(episode_count, total_steps,
                           current_episode_return,
                           cur_avg_dict, logger)

    def write_log(self, episode_count, total_steps, episode_return,
                  info_dict, logger=None):
        """Write log"""

        # ==
        # Aggregate
        self.exp_average_return = ((0.99 * self.exp_average_return)
                                   + (0.01 * episode_return))
        self.log_interval_sum_return += episode_return
        self.log_interval_sum_dict = add_log_dict(self.log_interval_sum_dict,
                                                  info_dict)

        # ==
        # Write at interval
        if episode_count % self.log_interval_episodes == 0:
            log_dict = avg_log_dict(self.log_interval_sum_dict,
                                    self.log_interval_episodes)
            # Incorporate into dict
            log_dict['episode_count'] = episode_count
            log_dict['total_steps'] = total_steps
            log_dict['episode_return'] = (self.log_interval_sum_return /
                                          self.log_interval_episodes)
            log_dict['exp_average_return'] = self.exp_average_return

            # Construct current episode log
            logStructDict = dataclasses.asdict(LogTupStruct(**log_dict))
            log_str = '|'.join([str(logStructDict[k]) for k in logStructDict])

            # Write or print
            if logger is not None:
                logger.info(log_str)
            else:
                print(log_str)

            # Clear
            self.log_interval_sum_return = 0
            self.log_interval_sum_dict = None
