# ============================================================================
# Runner class
#
# Inspired by a combination of runner and sampler classes from rlpyt,
# simplified for the incremental online setting. Adopted from the MinAtar
# training script.
#
# Author: Anthony G Chen
# ============================================================================
from collections import namedtuple, deque
import dataclasses
import json
import logging
import os
import time

import torch
import torch.multiprocessing as mp

# ==
# NOTE: need to do this for multiprocess to work with torch
try:
    mp.set_start_method('forkserver')  # forkserver or spawn
except RuntimeError:
    pass


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
    if new_dict is None:
        return sum_dict

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
TransitionTuple = namedtuple(
    'transition',
    'state, next_state, action, reward, is_terminal'
)


# ==
# Runner objects

class BaseRunner(object):
    """
    Base runner object
    """

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

    def world_dynamics(self, s, env, algo, total_steps):
        # network(s)[0] specifies the policy network, which we use to draw an action according to a multinomial
        # distribution over axis 1, (axis 0 iterates over samples, and is unused in this case. torch._no_grad()
        # avoids tracking history in autograd.
        with torch.no_grad():
            action = algo.get_action(s, total_steps).to(self.device)

        # Act according to the action and observe the transition and reward
        reward, terminated = env.act(action)
        reward = torch.tensor([[reward]], device=self.device).float()
        terminated = torch.tensor([[terminated]], device=self.device)

        # Obtain s_prime
        s_prime = self.get_state(env.state())

        return s_prime, action, reward, terminated

    def one_training_step(self, sample, total_steps):
        """
        One training step
        """
        raise NotImplementedError

    def train(self, n_steps):
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
        algo = self.algo  # pass reference

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
            algo.episode_reset()  # clear elig traces, etc.

            is_terminated = False

            while (not is_terminated) and total_steps < n_steps:
                # Generate data
                s_prime, action, reward, is_terminated = self.world_dynamics(
                    s, env, algo, total_steps,
                )

                sample = TransitionTuple(s, s_prime, action, reward, is_terminated)

                out_dict = self.one_training_step(sample, total_steps)

                # Accumulate
                current_episode_return += reward.item()
                current_episode_steps += 1
                total_steps += 1
                cur_sum_dict = add_log_dict(cur_sum_dict, out_dict)

                # Continue the process
                s = s_prime

            # Increment the episodes
            episode_count += 1

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


class IncrementalOnlineRunner(BaseRunner):
    def __init__(self, algo, EnvCls, env_kwargs,
                 log_interval_episodes=10,
                 log_dir_path=None,
                 store_checkpoint=False,
                 checkpoint_type='interval',
                 checkpoint_freq=2000,
                 checkpoint_dir_path='./checkpoints/',
                 load_model_path=None,
                 device='cpu'):
        super().__init__(
            algo, EnvCls, env_kwargs,
            log_interval_episodes=log_interval_episodes,
            log_dir_path=log_dir_path,
            store_checkpoint=store_checkpoint,
            checkpoint_type=checkpoint_type,
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir_path=checkpoint_dir_path,
            load_model_path=load_model_path,
            device=device,
        )

    def one_training_step(self, sample, total_steps):
        out_dict = self.algo.optimize_agent(sample, total_steps)
        return out_dict


class BatchedOfflineRunner(BaseRunner):
    def __init__(self, algo, EnvCls, env_kwargs,
                 BufferCls, buffer_kwargs,
                 batch_size=32,
                 replay_start_buffer_size=5000,
                 train_every_n_frames=1,
                 train_iterations=1,
                 log_interval_episodes=10,
                 log_dir_path=None,
                 store_checkpoint=False,
                 checkpoint_type='interval',
                 checkpoint_freq=2000,
                 checkpoint_dir_path='./checkpoints/',
                 load_model_path=None,
                 device='cpu'):
        super().__init__(
            algo, EnvCls, env_kwargs,
            log_interval_episodes=log_interval_episodes,
            log_dir_path=log_dir_path,
            store_checkpoint=store_checkpoint,
            checkpoint_type=checkpoint_type,
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir_path=checkpoint_dir_path,
            load_model_path=load_model_path,
            device=device,
        )

        self.replay_buffer = BufferCls(**buffer_kwargs)
        self.replay_start_buffer_size = replay_start_buffer_size
        self.batch_size = batch_size
        self.train_every_n_frames = train_every_n_frames
        self.train_iterations = train_iterations

    def one_training_step(self, sample, total_steps):
        # ==
        # Add experience to replay buffer
        self.replay_buffer.add(sample)

        out_dict = {}
        if total_steps % self.train_every_n_frames == 0:
            # ==
            # Sample from buffer
            batch_sample = None
            cur_buffer_size = self.replay_buffer.get_size()
            if ((cur_buffer_size > self.replay_start_buffer_size)
                    and (cur_buffer_size > self.batch_size)):
                for _ in range(self.train_iterations):
                    sample_batch = self.replay_buffer.sample(
                        self.batch_size
                    )  # (batch_n, tuple)
                    batch_sample = TransitionTuple(
                        *[torch.cat(e) for e in zip(*sample_batch)]
                    )  # (tuple key: torch.tensor of (batch_n, *)

            # ==
            # Train
            if batch_sample is not None:
                out_dict = self.algo.optimize_agent(batch_sample, total_steps)

        return out_dict


class AsynchBatchedOfflineRunner(BaseRunner):
    def __init__(self, algo, EnvCls, env_kwargs,
                 BufferCls, buffer_kwargs,
                 batch_size=32,  # dummy
                 replay_start_buffer_size=5000,
                 train_every_n_frames=1,
                 train_iterations=1,
                 log_interval_episodes=10,
                 log_dir_path=None,
                 store_checkpoint=False,
                 checkpoint_type='interval',
                 checkpoint_freq=2000,
                 checkpoint_dir_path='./checkpoints/',
                 load_model_path=None,
                 device='cpu'):
        super().__init__(
            algo, EnvCls, env_kwargs,
            log_interval_episodes=log_interval_episodes,
            log_dir_path=log_dir_path,
            store_checkpoint=store_checkpoint,
            checkpoint_type=checkpoint_type,
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir_path=checkpoint_dir_path,
            load_model_path=load_model_path,
            device=device,
        )

        self.BufferCls = BufferCls
        self.buffer_kwargs = buffer_kwargs
        self.replay_start_buffer_size = replay_start_buffer_size
        self.train_every_n_frames = train_every_n_frames
        self.train_iterations = train_iterations

    def train(self, n_steps):
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
        # Initialize processes

        lock = mp.Lock()
        self.algo.lock = lock

        sampler = AsynchSampler(
            actor=self.algo.actor,
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            lock=lock,
            device=self.device,
            train_every_n_frames=self.train_every_n_frames,
        )

        replay_buffer = self.BufferCls(**self.buffer_kwargs)

        # ==
        # Helper for sampling from buffer
        def buffer_sample():
            cur_buffer_size = replay_buffer.get_size()
            if ((cur_buffer_size > self.replay_start_buffer_size)
                    and (cur_buffer_size > self.buffer_kwargs['batch_size'])):
                sample_batch = replay_buffer.sample()  # (batch_n, iter)
                if sample_batch is not None:
                    batch_sample = TransitionTuple(
                        *[torch.cat(e) for e in zip(*sample_batch)]
                    )  # (tuple key: torch.tensor of (batch_n, *)
                    return batch_sample
            return None

        # ==
        # Counter variables and iterate training
        step_counts = 0
        episode_count = 0

        current_episode_steps = 0
        current_episode_return = 0.0

        while step_counts < n_steps:

            # ==
            # Sample environment for transitions
            transitions_lists = sampler.step()  # (train_every_n_frames, ) dicts

            # ==
            # Record statistics and store to buffer
            for tranList in transitions_lists:
                # ==
                # Process statistics
                # List entries: state, next_state, action, reward, is_terminal
                current_episode_steps += 1
                current_episode_return += tranList[3].item()  # reward

                if tranList[4].item():  # if done
                    episode_count += 1

                    self.write_log(episode_count, step_counts,
                                   current_episode_return,
                                   info_dict={})

                    current_episode_steps = 0
                    current_episode_return = 0.0

                step_counts += 1
                replay_buffer.add(tranList)

            # ==
            # Train
            if step_counts >= self.replay_start_buffer_size:
                for _ in range(self.train_iterations):
                    # Sample
                    batch_sample = buffer_sample()

                    # Train network
                    if batch_sample is not None:
                        with lock:
                            out_dict = self.algo.optimize_agent(batch_sample, step_counts)

        sampler.close()
        replay_buffer.close()


class AsynchSampler(mp.Process):
    """
    Asynchronous sampler for actor-environment interaction samples
    Inspired by
    https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/BaseAgent.py
    """
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, actor, EnvCls, env_kwargs, lock, device,
                 train_every_n_frames=1):
        mp.Process.__init__(self)
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._actor = actor
        self.EnvCls = EnvCls
        self.env_kwargs = env_kwargs
        self.device = device
        self.train_every_n_frames = train_every_n_frames

        self._env = self.EnvCls(**self.env_kwargs)

        self._state = None
        self._total_steps = 0
        self.__cache_len = 2

        self.lock = lock

        self.start()  # start process

    def _sample(self):
        transitions = []
        for _ in range(self.train_every_n_frames):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def _transition(self):
        """
        Environment-action loop
        :return: namedtuple of individual transitions
        """

        def cast_state(s):
            tensor_s = torch.tensor(s, device=self.device)
            tensor_s = tensor_s.permute(2, 0, 1).unsqueeze(0).float()
            return tensor_s

        # Start of episode
        if self._state is None:
            self._env.reset()
            self._state = cast_state(self._env.state())
            # algo.episode_reset()   # NOTE maybe add some version of this

        # Take action
        with self.lock:
            with torch.no_grad():
                # NOTE maybe TODO: potential speed-up with using only network here
                # rather than using the agent rng for sampling while locked
                t_action = self._actor.get_action(self._state, self._total_steps)

        # Get environment outcomes (MinAtar specific)
        action = t_action.item()
        reward, terminated = self._env.act(action)

        t_reward = torch.tensor([[reward]], device=self.device).float()
        t_terminated = torch.tensor([[terminated]], device=self.device)
        s_prime = cast_state(self._env.state())

        # Get transition
        # Order: 'state', 'next_state', 'action', 'reward', 'is_terminal'
        transition = [self._state, s_prime, t_action, t_reward, t_terminated]

        # Increment
        self._total_steps += 1
        self._state = None if terminated else s_prime.clone()

        return transition

    def _set_up(self):
        pass

    def run(self):
        self._set_up()
        self._env = self.EnvCls(**self.env_kwargs)

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._actor.set_model(data)
            else:
                raise NotImplementedError

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._actor.set_model(net)
        else:
            self.__pipe.send([self.NETWORK, net])
