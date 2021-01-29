# ============================================================================
# Runner class
#
# Inspired by a combination of runner and sampler classes from rlpyt,
# simplified for the incremental online setting. Adopted from the MinAtar
# training script.
#
# Author: Anthony G Chen
# ============================================================================
from collections import namedtuple
import dataclasses
import json
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
    lsf_v_v_diff: float = None
    et_loss: float = None


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
                 store_checkpoint=False,
                 checkpoint_type='interval',
                 checkpoint_freq=2000,
                 checkpoint_dir_path='./checkpoints/',
                 load_model_path=None,
                 device='cpu'):

        self.algo = algo
        self.EnvCls = EnvCls
        self.env_kwargs = env_kwargs
        self.device = device

        self.log_interval_episodes = log_interval_episodes
        self.log_dir_path = log_dir_path
        self.logger = None
        self.log_delim_str = '|'

        self.store_checkpoint = store_checkpoint
        self.checkpoint_type = checkpoint_type
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir_path = checkpoint_dir_path

        self.load_model_path = load_model_path  # load from checkpoint

        # ==
        # Initialize
        print('Initializing environment...')
        self.environment = self.EnvCls(**self.env_kwargs)
        print(self.environment)

        print('Initializing algo...')
        self.algo.initialize(self.environment, self.device)
        # Possibly load from checkpoint
        if self.load_model_path is not None:
            print(f'Loading model from: {self.load_model_path}')
            ckpt_model = torch.load(self.load_model_path)
            self.algo.model.load_state_dict(ckpt_model)
            self.algo.model = self.algo.model.to(self.device)
        print(self.algo)

        # ==
        # Counter items
        self.exp_average_return = 0.0
        self.log_interval_sum_return = 0.0
        self.log_interval_sum_dict = None

    def init_logger(self, logger_path,
                    logger_name='Experiment') -> logging.Logger:
        """
        Initializes the path to write the log to
        :param logger_name: name of log
        :return: logging.Logger object
        """
        self.logger = logging.getLogger(logger_name)

        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(logger_path)  # TOD bug here
        formatter = logging.Formatter('%(asctime)s||%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

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
        log_title_str = self.log_delim_str.join(dc_fields)
        if self.log_dir_path is not None:
            log_path = os.path.join(self.log_dir_path, 'progress.csv')
            self.init_logger(log_path)
            self.logger.info(log_title_str)
        else:
            print(log_title_str)

        # ==
        # Initialize
        episode_count = 0
        total_steps = 0

        env = self.environment
        network = self.algo.model  # pass reference

        # ==
        # Run training
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

                # Accumulate
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
                           cur_avg_dict)

            # ==
            # Potentially checkpoint
            self.save_checkpoint(episode_count, total_steps,
                                 current_episode_return,
                                 cur_avg_dict)

        # ==
        # Post-training checkpoint
        self.save_checkpoint(episode_count, total_steps,
                             current_episode_return,
                             cur_avg_dict, force=True)

    def write_log(self, episode_count, total_steps, episode_return,
                  info_dict):
        """Write log (at interval)"""

        # ==
        # Aggregate
        self.exp_average_return = ((0.99 * self.exp_average_return)
                                   + (0.01 * episode_return))
        self.log_interval_sum_return += episode_return
        self.log_interval_sum_dict = add_log_dict(self.log_interval_sum_dict,
                                                  info_dict)

        # ==
        # Write at interval
        # TODO: should also do a write at the last step
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
            log_str = self.log_delim_str.join([str(logStructDict[k])
                                               for k in logStructDict])

            # Write or print
            if self.logger is not None:
                self.logger.info(log_str)
            else:
                print(log_str)

            # Clear
            self.log_interval_sum_return = 0
            self.log_interval_sum_dict = None

    def save_checkpoint(self, episode_count, total_steps,
                        episode_return, info_dict,
                        force=False):
        # Conditions
        if (force or (self.store_checkpoint
                      and episode_count % self.checkpoint_freq == 0)):

            # ==
            # Check dir
            if not os.path.exists(self.checkpoint_dir_path):
                os.makedirs(self.checkpoint_dir_path)

            # ==
            # Make name
            if self.checkpoint_type == 'last':
                out_name = f'ckpt'
            elif self.checkpoint_type == 'interval':
                out_name = f'cpkt_steps-{total_steps}_epis-{episode_count}'
            else:
                raise NotImplementedError
            out_path = os.path.join(self.checkpoint_dir_path, out_name)

            # ==
            # Data and save
            out_json = {
                'episode_count': episode_count,
                'total_steps': total_steps,
                'current_episode_return': episode_return,
                'cur_avg_dict': info_dict,
            }

            torch.save(obj=self.algo.model.state_dict(),
                       f=f'{out_path}.pt')
            with open(f'{out_path}.json', 'w') as fp:
                json.dump(out_json, fp)

