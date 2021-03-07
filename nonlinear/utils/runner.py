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
except RuntimeError as e:
    print('mp RuntimeError:', e)
    pass

# ==
# Logging related
RunnerTupStruct = namedtuple(
    'Runner',
    field_names=['Diagnostic_env_name', 'Diagnostic_algo_name',
                 'Diagnostic_episode_count', 'Diagnostic_step_count',
                 'Diagnostic_sec_elapsed', 'Diagnostic_steps_per_sec',
                 'Diagnostic_exp_average_return',
                 'episode_return', 'episode_steps']
)


def aggregate_log_dict(agg_dict, new_dict) -> dict:
    """
    Aggregate the statistics of a log dict
    :param agg_dict: aggregation dictionary
    :param new_dict: dict with new stats
    :return: new aggregation dict with aggregated
    """
    for k in new_dict:
        # init new if not present
        if k not in agg_dict:
            agg_dict[k] = {
                'n': 0,
                'sum': 0.0,
                'max': new_dict[k],
                'min': new_dict[k],
            }
        # aggregate
        agg_dict[k]['n'] += 1
        agg_dict[k]['sum'] += new_dict[k]
        agg_dict[k]['max'] = max(new_dict[k], agg_dict[k]['max'])
        agg_dict[k]['min'] = min(new_dict[k], agg_dict[k]['min'])
        # TODO: add more stats (e.g. stdev, max, minin the future)

    return agg_dict


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
        self.log_header_written = False

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
        if self.log_dir_path is not None:
            log_path = os.path.join(self.log_dir_path, 'progress.csv')
            self.init_logger(log_path)
        else:
            self.logger = None

        # ==
        # Initialize
        total_step_counts = 0
        total_epis_counts = 0

        t0 = time.time()
        t_interval_steps = 0

        # episode_count = 0
        # total_steps = 0

        cur_agg_dict = {}  # for logging per-step and per-epis stats

        env = self.environment
        algo = self.algo  # pass reference

        # ==
        # Run training
        while total_step_counts < n_steps:
            # ==
            # Initialize environment and start state
            current_episode_steps = 0
            current_episode_return = 0.0

            env.reset()
            s = self.get_state(env.state())
            algo.episode_reset()  # clear elig traces, etc.

            is_terminated = False

            while (not is_terminated) and total_step_counts < n_steps:
                # Generate data
                s_prime, action, reward, is_terminated = self.world_dynamics(
                    s, env, algo, total_step_counts,
                )

                sample = [s, s_prime, action, reward, is_terminated]

                out_dict = self.one_training_step(sample, total_step_counts)
                cur_agg_dict = aggregate_log_dict(
                    cur_agg_dict, out_dict
                )  # aggregate stats

                # Accumulate
                current_episode_return += reward.item()
                current_episode_steps += 1
                total_step_counts += 1
                t_interval_steps += 1

                # Continue the process
                s = s_prime

            # Increment the episodes
            total_epis_counts += 1
            cur_agg_dict = aggregate_log_dict(
                cur_agg_dict,
                {'episode_return': current_episode_return,
                 'episode_steps': current_episode_steps}
            )

            # ==
            # Write logs
            if total_epis_counts % self.log_interval_episodes == 0:
                # Generate info dict to write
                t_elapse = time.time() - t0
                steps_ps = t_interval_steps / t_elapse
                info_dict = {
                    'Diagnostic/episode_count': total_epis_counts,
                    'Diagnostic/step_count': total_step_counts,
                    'Diagnostic/sec_elapsed': t_elapse,
                    'Diagnostic/steps_per_sec': steps_ps,
                }

                # Write
                self.write_log(info_dict, cur_agg_dict)

                # Per log reset
                t0 = time.time()
                t_interval_steps = 0
                cur_agg_dict = {}

            # ==
            # Potentially checkpoint
            if (self.store_checkpoint and
                    total_epis_counts % self.checkpoint_freq == 0):
                self.save_checkpoint(total_epis_counts, total_step_counts,
                                     current_episode_return,
                                     info_dict={})

        # ==
        # Post-training checkpoint
        self.save_checkpoint(total_epis_counts, total_step_counts,
                             current_episode_return,
                             info_dict={})

    def write_log(self,
                  info_dict: dict,
                  aggregate_dict: dict):
        """Write log"""

        # Helpers
        stat_list = ['Avg', 'Min', 'Max']

        def construct_out_logfields(*tups):
            """Helper method that takes in a variable number of namedtuples
            and construct a joint namedtuple for write-out"""
            fields = []

            for tupStruct in tups:
                for k in tupStruct._fields:
                    if k.startswith("Diagnostic"):
                        k_str = k.replace('_', '/', 1)
                        fields.append(k_str)
                    else:
                        for st in stat_list:
                            fields.append(f'{k}/{st}')

            return fields

        def process_agg_dict(agg_dict):
            out_dict = {}
            for k in agg_dict:
                n = agg_dict[k]['n']
                out_dict[f'{k}/Avg'] = agg_dict[k]['sum'] / n
                out_dict[f'{k}/Min'] = agg_dict[k]['min']
                out_dict[f'{k}/Max'] = agg_dict[k]['max']
            return out_dict

        # Process the aggregate dict
        all_info_dict = {
            'Diagnostic/env_name': ''.join([
                self.env_kwargs[k] for k in self.env_kwargs if 'name' in k
            ]),
            'Diagnostic/algo_name': type(self.algo).__name__,
            **info_dict,
            **process_agg_dict(aggregate_dict),
        }
        jointTupFields = construct_out_logfields(
            RunnerTupStruct, self.algo.logTupStruct
        )

        # ==
        # Write header if have not written yet (once per runner instance)
        if not self.log_header_written:
            log_title_str = self.log_delim_str.join(jointTupFields)
            self.logger.info(log_title_str)
            self.log_header_written = True

        # ==
        # Write all entries
        log_list = []
        for f in jointTupFields:
            if f in all_info_dict:
                log_list.append(str(all_info_dict[f]))
            else:
                log_list.append(str(None))
        log_str = self.log_delim_str.join(log_list)
        self.logger.info(log_str)

    def save_checkpoint(self, episode_count, total_steps,
                        episode_return, info_dict):
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
            'info': info_dict,
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
                    samples = self.replay_buffer.sample(
                        self.batch_size
                    )  # (batch_n, iter)
                    batch_sample = TransitionTuple(*samples)

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
        if self.log_dir_path is not None:
            log_path = os.path.join(self.log_dir_path, 'progress.csv')
            self.init_logger(log_path)
        else:
            self.logger = None

        # ==
        # Initialize processes
        lock = mp.Lock()
        self.algo.lock = lock
        self.algo.actor.model.share_memory()

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
        # Counter variables and iterate training
        total_step_counts = 0
        total_epis_counts = 0
        current_episode_steps = 0
        current_episode_return = 0.0

        t0 = time.time()
        t_interval_steps = 0

        cur_agg_dict = {}  # for logging per-step and per-epis stats

        while total_step_counts < n_steps:

            # ==
            # Sample environment for transitions
            transitions_lists = sampler.step()  # (train_every_n_frames, ) list

            # ==
            # Record statistics and store to buffer
            for tranList in transitions_lists:
                # ==
                # Process statistics
                # List entries: state, next_state, action, reward, is_terminal
                total_step_counts += 1
                current_episode_steps += 1
                t_interval_steps += 1

                reward, done = tranList[3].item(), tranList[4].item()
                current_episode_return += reward

                # ==
                # If episode terminates
                if done:
                    # More aggregations
                    total_epis_counts += 1
                    cur_agg_dict = aggregate_log_dict(
                        cur_agg_dict,
                        {'episode_return': current_episode_return,
                         'episode_steps': current_episode_steps}
                    )

                    # Write logs at intervals
                    if total_epis_counts % self.log_interval_episodes == 0:
                        # Generate info dict to write
                        t_elapse = time.time() - t0
                        steps_ps = t_interval_steps / t_elapse
                        info_dict = {
                            'Diagnostic/episode_count': total_epis_counts,
                            'Diagnostic/step_count': total_step_counts,
                            'Diagnostic/sec_elapsed': t_elapse,
                            'Diagnostic/steps_per_sec': steps_ps,
                        }

                        # Write
                        self.write_log(info_dict, cur_agg_dict)

                        # Per log reset
                        t0 = time.time()
                        t_interval_steps = 0
                        cur_agg_dict = {}

                    # Per episode reset
                    current_episode_steps = 0
                    current_episode_return = 0.0

                # ==
                # Add transition to buffer
                replay_buffer.add(tranList)

            # ==
            # Train via sampling from buffer
            if total_step_counts >= self.replay_start_buffer_size:
                for _ in range(self.train_iterations):
                    # ==
                    # Sample
                    batch_sample = None
                    cur_buffer_size = replay_buffer.get_size()
                    if ((cur_buffer_size > self.replay_start_buffer_size)
                            and (cur_buffer_size > self.buffer_kwargs['batch_size'])):
                        samples = replay_buffer.sample()  # (batch_n, iter)
                        if samples is not None:
                            batch_sample = TransitionTuple(*samples)

                    # ==
                    # Train network
                    if batch_sample is not None:
                        loss = self.algo.compute_loss(batch_sample)
                        self.algo.optimizer.zero_grad()
                        loss.backward()
                        with lock:
                            self.algo.optimizer.step()
                        out_dict = self.algo.post_update_step(loss)
                        cur_agg_dict = aggregate_log_dict(cur_agg_dict, out_dict)
                        # TODO: do the set-network explicitly here?

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
        # NOTE maybe: potential speed-up with using only network here
        # with self.lock:  # TODO NOTE is this needed?
        with torch.no_grad():
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
        data = self.__pipe.recv()
        data = [[x.clone() for x in tranList] for tranList in data]
        # TODO future: need a way to send back information to be logged as well
        return data

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._actor.set_model(net)
        else:
            self.__pipe.send([self.NETWORK, net])
