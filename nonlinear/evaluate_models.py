# ============================================================================
# Evaluation script for saved models
# ============================================================================

from collections import namedtuple
import glob
import json
import logging
import os
import time
import yaml

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import random
import torch

from minatar import Environment  # MinAtar repo

from models.ac_network import ACNetwork
from models.lsf_ac_network import LSF_ACNetwork
from models.q_network import QNetwork
from models.lsf_q_network import *

from utils.runner import aggregate_log_dict
from utils.replay_buffer import SimpleReplayBuffer

# ==========
# Logging related

EvalTupStruct = namedtuple(
    'Eval',
    field_names=[
        'Checkpoint_episode_count', 'Checkpoint_total_steps',
        'Diagnostic_env_name', 'Diagnostic_algo_cls_string',
        'Diagnostic_model_cls_string',
        'Diagnostic_algo_info', 'Diagnostic_sf_lambda',
        'Diagnostic_episode_count', 'Diagnostic_step_count',
        'Diagnostic_sec_elapsed', 'Diagnostic_steps_per_sec',
        'episode_return', 'episode_steps',
        'phi_norm', 'pred_reward', 'reward_mse',
        'D_srank', 'D_norm',  # from on-policy distr
    ]
)


def construct_out_logfields(*tups):
    """Helper method that takes in a variable number of namedtuples
    and construct a joint namedtuple for write-out"""
    stat_list = ['Avg', 'Min', 'Max']

    fields = []
    for tupStruct in tups:
        for k in tupStruct._fields:
            if k.startswith("Diagnostic"):
                k_str = k.replace('_', '/', 1)
                fields.append(k_str)
            elif k.startswith("Checkpoint"):
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


# ==========
# Running things

def per_step_evaluation(actor, sample_list):
    cur_s, nex_s, action, reward, is_terminated = sample_list
    model = actor.model

    # phi
    with torch.no_grad():
        # phi
        phi = model.encoder(cur_s)
        phi_norm = torch.norm(phi)

        # reward error
        pred_r = model.reward_fn(phi)
        rew_mse = (pred_r - reward) ** 2

    #
    out_dict = {
        'phi_norm': phi_norm.item(),
        'pred_reward': pred_r.item(),
        'reward_mse': rew_mse.item(),
    }

    return out_dict


def post_run_evaluation(actor, buffer):
    def compute_srank(phiMat):
        """Compute srank https://openreview.net/forum?id=O9bnihsFfXU"""
        thresh_delta = 0.01

        (U, S, V) = torch.svd(phiMat, compute_uv=False)
        sVec = S.numpy()
        sv_sum = np.sum(sVec)

        sv_cumu = 0.0
        k = 0
        while (sv_cumu / sv_sum) < (1. - thresh_delta):
            sv_cumu += sVec[k]
            k += 1

        return k

    model = actor.model
    batch_sample = buffer.sample(batch_size=2048)
    [cur_s, nex_s, action, reward, is_terminated] = batch_sample

    with torch.no_grad():
        # phi
        phi = model.encoder(cur_s)
        srank = compute_srank(phi)
        norm = np.average(torch.norm(phi).numpy())

    out_dict = {
        'D_srank': srank,
        'D_norm': norm,
    }

    return out_dict


def run_eval_generate_log(env, actor,
                          general_info: dict,
                          cfg: DictConfig = None,
                          logger=None):
    device = 'cpu'
    max_steps = cfg.eval.max_steps
    buffer_size = cfg.eval.buffer_size

    def get_state(state):
        return (torch.tensor(state, device=device).permute(2, 0, 1)
                ).unsqueeze(0).float()

    def world_dynamics(state, envr, actr):
        with torch.no_grad():
            action = actr.get_action(state).to(device)

        # Act according to the action and observe the transition and reward
        reward, terminated = envr.act(action)
        reward = torch.tensor([[reward]], device=device).float()
        terminated = torch.tensor([[terminated]], device=device)

        # Obtain s_prime
        state_prime = get_state(env.state())

        return state_prime, action, reward, terminated

    # ==
    # Run environment
    replay_buffer = SimpleReplayBuffer(
        buffer_size=buffer_size,
        seed=0
    )

    total_step_counts = 0
    total_epis_counts = 1
    t0 = time.time()
    t_interval_steps = 0

    while total_step_counts <= max_steps:
        cur_agg_dict = {}  # for logging per-step stats
        current_episode_steps = 0
        current_episode_return = 0.0

        env.reset()
        cur_s = get_state(env.state())

        is_terminated = False
        while not is_terminated:
            # Interact
            nex_s, action, reward, is_terminated = world_dynamics(
                cur_s, env, actor,
            )

            # Generate information to log
            sample = [cur_s, nex_s, action, reward, is_terminated]
            replay_buffer.add(sample)

            out_dict = per_step_evaluation(actor, sample)
            cur_agg_dict = aggregate_log_dict(
                cur_agg_dict,
                out_dict,
            )

            # Aggregate and increment
            current_episode_return += reward.item()
            current_episode_steps += 1
            total_step_counts += 1
            t_interval_steps += 1
            cur_s = nex_s

        # Aggregate
        cur_agg_dict = aggregate_log_dict(
            cur_agg_dict,
            {'episode_return': current_episode_return,
             'episode_steps': current_episode_steps}
        )

        t_elapse = time.time() - t0
        steps_ps = t_interval_steps / t_elapse
        cur_info_dict = {
            'Diagnostic/episode_count': total_epis_counts,
            'Diagnostic/step_count': total_step_counts,
            'Diagnostic/sec_elapsed': t_elapse,
            'Diagnostic/steps_per_sec': steps_ps,
            **general_info,
        }

        # Write log and reset
        write_log(info_dict=cur_info_dict,
                  aggregate_dict=cur_agg_dict,
                  logger=logger)
        t0 = time.time()
        t_interval_steps = 0
        total_epis_counts += 1

    # ==
    # Post-run analysis
    cur_agg_dict = {}
    for __ in range(8):
        out_dict = post_run_evaluation(actor=actor,
                                       buffer=replay_buffer)
        cur_agg_dict = aggregate_log_dict(cur_agg_dict, out_dict)
    cur_info_dict = {
        'Diagnostic/episode_count': total_epis_counts,
        'Diagnostic/step_count': total_step_counts,
        **general_info,
    }
    write_log(info_dict=cur_info_dict,
              aggregate_dict=cur_agg_dict,
              logger=logger)


def write_log(info_dict: dict,
              aggregate_dict: dict,
              logger=None):
    """
    Log writer. Copied / adapted from utils.runner.BaseRunner
    """
    # Process the aggregate dict
    all_info_dict = {
        **info_dict,
        **process_agg_dict(aggregate_dict),
    }
    jointTupFields = construct_out_logfields(
        EvalTupStruct,
    )

    # ==
    # Write all entries
    log_list = []
    for f in jointTupFields:
        if f in all_info_dict:
            log_list.append(str(all_info_dict[f]))
        else:
            log_list.append(str(None))
    log_str = '|'.join(log_list)
    if logger is not None:
        logger.info(log_str)
    else:
        print(log_str)


def initialize_and_run_saved_models(model_path: str,
                                    config_path: str,
                                    ckpt_json_path: str,
                                    cfg: DictConfig = None,
                                    logger=None):
    """
    Initialize from saved files and run model
    :param model_path: path to the saved model .pt file
    :param config_path: path to the hydra config file
    :param ckpt_json_path: path to the saved model .json file
    :param logger: logger instance
    """
    # ==
    # Read configuration and information files

    # Read hydra config file
    with open(config_path, 'r') as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Read checkpoint information json file
    with open(ckpt_json_path, 'r') as stream:
        ckpt_info = json.load(stream)

    # Print?
    # print('config_dict', type(cfg_dict), cfg_dict)

    # Set up relevant information for logging
    algo_info_str = str({
        'discount_gamma': cfg_dict['algo']['kwargs']['discount_gamma'],
    })
    general_info_dict = {
        'Checkpoint/episode_count': ckpt_info['episode_count'],
        'Checkpoint/total_steps': ckpt_info['total_steps'],
        'Diagnostic/env_name': cfg_dict['env']['kwargs']['env_name'],
        'Diagnostic/algo_cls_string': cfg_dict['algo']['cls_string'],
        'Diagnostic/model_cls_string': cfg_dict['model']['cls_string'],
        'Diagnostic/algo_info': algo_info_str,
        'Diagnostic/sf_lambda': cfg_dict['algo']['kwargs']['sf_lambda'],
    }

    # ==
    # Initialize environment and agent model
    env_cls = Environment  # MinAtar env
    env_kwargs = {
        'random_seed': cfg_dict['training']['seed'],
        **cfg_dict['env']['kwargs']
    }
    env = env_cls(**env_kwargs)

    model_cls = globals()[cfg_dict['model']['cls_string']]
    model_kwargs = cfg_dict['model']['kwargs']
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    model = model_cls(
        in_channels, num_actions,
        **model_kwargs
    )

    ckpt_model = torch.load(model_path)
    model.load_state_dict(ckpt_model)

    actor = Actor(model=model, num_actions=num_actions,
                  algo_cls_string=cfg_dict['algo']['cls_string'],
                  epsilon=cfg.actor.epsilon,
                  sf_lambda=cfg_dict['algo']['kwargs']['sf_lambda'],
                  device='cpu',
                  seed=cfg_dict['training']['seed'])

    # ==
    # Run in environment and generate information
    run_eval_generate_log(env=env, actor=actor,
                          general_info=general_info_dict,
                          cfg=cfg,
                          logger=logger)
    # NOTE: not writing a EvalRunner because I want to share loggers across
    #       multiple saved models


class Actor:
    """
    Default generic actor for taking actions in the environment
    """

    def __init__(self, model, num_actions,
                 algo_cls_string,
                 epsilon,
                 sf_lambda,
                 device='cpu',
                 seed=0):
        self.model = model  # policy network
        self.device = device
        self.algo_cls_string = algo_cls_string

        self.num_actions = num_actions
        self.sf_lambda = sf_lambda
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

    def get_action(self, state):
        """
        Get action with epsilon greedy policy
        :param state:
        :return:
        """

        if self.rng.binomial(1, self.epsilon) == 1:
            # If epsilon uniform random action
            action = torch.tensor([[self.rng.integers(self.num_actions)]],
                                  device=self.device)
        else:
            # Greedy action
            with torch.no_grad():
                # NOTE: might need to check device, but I should just keep
                #       everything on CPU
                if self.algo_cls_string == 'LSF_DQN':
                    action = self.model(state, self.sf_lambda).max(1)[1].view(1, 1)
                else:
                    raise NotImplementedError

        return action


@hydra.main(config_path="conf/eval", config_name="eval_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # print the configs

    dir_wc = cfg.path.dir_wc
    dir_paths = glob.glob(dir_wc)

    model_append_wc = cfg.path.model_append_wc
    config_append_wc = cfg.path.config_append_wc

    #
    # ==
    # Initialize logger and print first line
    logger_name = 'Evaluation'
    logger_path = 'outcome_eval.csv'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logger_path)
    formatter = logging.Formatter('%(asctime)s||%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    jointTupFields = construct_out_logfields(
        EvalTupStruct)
    log_title_str = '|'.join(jointTupFields)
    if logger is not None:
        logger.info(log_title_str)
    else:
        print(log_title_str)

    # ==
    # Run
    for dpath in dir_paths:
        model_wc = os.path.join(dpath, model_append_wc)
        config_wc = os.path.join(dpath, config_append_wc)

        model_paths = glob.glob(model_wc)
        cjson_paths = [
            p.replace('.pt', '.json') for p in model_paths
        ]
        config_paths = glob.glob(config_wc)

        assert len(model_paths) == len(cjson_paths)
        assert len(config_paths) == 1

        for i in range(len(model_paths)):
            cur_model_path = model_paths[i]
            cur_cjson_path = cjson_paths[i]
            cur_config_path = config_paths[0]

            print('model:', cur_model_path,
                  'json:', cur_cjson_path,
                  'config', cur_config_path)

            initialize_and_run_saved_models(
                model_path=cur_model_path,
                config_path=cur_config_path,
                ckpt_json_path=cur_cjson_path,
                cfg=cfg,
                logger=logger,
            )

    """
    seed = ??
    if seed is None:
        seed = 0
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """


if __name__ == "__main__":
    main()
