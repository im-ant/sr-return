# =============================================================================
# Training the linear agent on linear prediction environments
#
# Author: Anthony G. Chen
# =============================================================================

import argparse
import configparser
import dataclasses
import logging
import os

import gym
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

from algos.sf_return_ag import SFReturnAgent
from algos.td_lambda_ag import SarsaLambdaAgent
from algos.expt_trace_ag import ExpectedTraceAgent
from envs.boyans_chain import BoyansChainEnv
from envs.random_walk_chain import RandomWalkChainEnv
from envs.perf_bin_tree import PerfBinaryTreeEnv
import utils.mdp_utils as mut


@dataclasses.dataclass
class LogTupStruct:
    num_episodes: int = None
    envCls_name: str = None
    env_kwargs: str = None
    agentCls_name: str = None
    seed: int = None
    gamma: float = None
    lr: float = None
    lamb: float = None
    eta_trace: float = None
    use_true_reward_params: bool = None
    use_true_sf_params: bool = None
    episode_idx: int = None  # episodic-specific logs
    total_steps: int = None
    cumulative_reward: float = None
    v_fn_rmse: float = None
    sf_G_rmse: float = None
    sf_matrix_rmse: float = None
    value_loss_avg: float = None  # agent log dict specific logs
    reward_loss_avg: float = None
    sf_loss_avg: float = None
    et_loss_avg: float = None


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


def _initialize_environment(cfg: DictConfig) -> object:
    """
    Helper function to initialize the environment object
    :param cfg:
    :return: gym.Env object
    """
    envCls = globals()[cfg.env.cls_string]  # hacky
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs)  # convert to dict
    env_kwargs['seed'] = cfg.training.seed  # add global seed
    environment = envCls(**env_kwargs)
    return environment


def _initialize_agent(cfg: DictConfig, environment) -> object:
    """
    Helper method to initialize an agent object
    :param cfg: hydra config dict
    :param environment: gym environment
    :return: agent object
    """

    # ==
    # Initialize with config parameters
    agentCls = globals()[cfg.agent.cls_string]  # hacky class
    agent_kwargs = OmegaConf.to_container(cfg.agent.kwargs)  # convert to dict
    agent_kwargs['seed'] = cfg.training.seed

    # Initialize and return
    agent = agentCls(
        feature_dim=environment.observation_space.shape[0],  # assume linear
        num_actions=environment.action_space.n,
        **agent_kwargs
    )

    # ==
    # (Optional) Initialize true reward and SF functions

    # Initialize solved SF parameters
    if hasattr(cfg.agent.kwargs, 'use_true_sf_params'):
        if cfg.agent.kwargs.use_true_sf_params:
            linear_sf_param = mut.solve_linear_sf_param(
                env=environment,
                gamma=(cfg.agent.kwargs.gamma * cfg.agent.kwargs.lamb)
            )
            agent.Ws[0] = linear_sf_param
    # Initialize solved reward parameters
    if hasattr(cfg.agent.kwargs, 'use_true_reward_params'):
        if cfg.agent.kwargs.use_true_reward_params:
            linear_reward_param = mut.solve_linear_reward_param(
                env=environment,
            )
            agent.Wr = linear_reward_param

    return agent


def _pre_compute(cfg: DictConfig, environment, agent) -> dict:
    """
    Pre-compute things before training begins, largely used to solve for the
    true value function to measure RMSE.

    :param cfg: config dict
    :param environment: enviornment object
    :param agent: agent object
    :return: dict of pre-computed items
    """
    # Compute the true MDP value function
    true_v_fn = mut.solve_value_fn(environment,
                                   cfg.agent.kwargs.gamma)

    # Compute the true MDP successor feature
    true_sf_mat = mut.solve_successor_feature(
        environment,
        (cfg.agent.kwargs.gamma * cfg.agent.kwargs.lamb)
    )

    out_dict = {
        'true_value_vec': true_v_fn,
        'true_sf_mat': true_sf_mat,
    }

    return out_dict


def _extract_agent_log_dict(agent):
    """
    Helper method to extract the agent's logged items for the logger
    :param agent: agent object
    :return: dictionary for epis_log_dict
    """
    # The output log dictionary
    # NOTE: need ot make this fit with the keys of LogTupStruct
    log_dict = {
        'value_loss_avg': None,
        'reward_loss_avg': None,
        'sf_loss_avg': None,
        'et_loss_avg': None,
    }

    # ==
    # Extract agent log
    ag_dict = agent.log_dict
    if ag_dict is None:
        return log_dict

    # Compute
    # TODO: add more conditions to check if the list is empty?
    if 'value_errors' in ag_dict:
        avg_value_loss = np.average(
            np.square(ag_dict['value_errors'])
        )
        log_dict['value_loss_avg'] = avg_value_loss

    if 'reward_errors' in ag_dict:
        avg_rew_loss = np.average(
            np.square(ag_dict['reward_errors'])
        )
        log_dict['reward_loss_avg'] = avg_rew_loss

    if 'sf_error_norms' in ag_dict:
        # NOTE: it is already the norm so positive
        avg_sf_loss = np.average(ag_dict['sf_error_norms'])
        log_dict['sf_loss_avg'] = avg_sf_loss

    if 'et_error_norms' in ag_dict:
        # NOTE: it is already the norm so positive
        avg_et_loss = np.average(ag_dict['et_error_norms'])
        log_dict['et_loss_avg'] = avg_et_loss

    return log_dict


def write_post_episode_log(cfg: DictConfig,
                           pre_comput_dict: dict,
                           episode_dict: dict,
                           environment, agent,
                           logger) -> None:
    log_dict = {
        'num_episodes': cfg.training.num_episodes,
        'envCls_name': cfg.env.cls_string,
        'env_kwargs': str(cfg.env.kwargs),
        'agentCls_name': cfg.agent.cls_string,
        'seed': cfg.training.seed,  # global seed
        'lr': cfg.agent.lr,
    }

    # Assume all agent kwargs are available in log
    for k in cfg.agent.kwargs:
        log_dict[k] = cfg.agent.kwargs[k]

    # ==
    # Learning RMSEs

    # For the value function
    log_dict['v_fn_rmse'] = mut.evaluate_value_rmse(
        environment, agent, pre_comput_dict['true_value_vec']
    )
    # For LSF value function
    log_dict['sf_G_rmse'] = mut.evaluate_sf_ret_rmse(
        environment, agent, pre_comput_dict['true_value_vec']
    )
    # (Optional) for SF matrix (NOTE: hard-coded)
    if cfg.agent.cls_string == 'SFReturnAgent':
        log_dict['sf_matrix_rmse'] = mut.evaluate_sf_mat_rmse(
            environment, agent, pre_comput_dict['true_sf_mat']
        )

    # ==
    # episode-specific logs

    log_dict['episode_idx'] = episode_dict['episode_idx']
    log_dict['total_steps'] = episode_dict['total_steps']
    log_dict['cumulative_reward'] = episode_dict['cumulative_reward']

    # ==
    # Get losses from the agent logs
    agent_log_dict = _extract_agent_log_dict(agent)
    for k in agent_log_dict:
        log_dict[k] = agent_log_dict[k]

    # ==
    # Write the output log

    # Construct log output
    logStructDict = dataclasses.asdict(LogTupStruct(**log_dict))
    log_str = '||'.join([str(logStructDict[k]) for k in logStructDict])
    # Write or print
    if logger is not None:
        logger.info(log_str)
    else:
        if (episode_idx + 1) % args.log_print_freq == 0:
            print(log_str)


def run_single_linear_experiment(cfg: DictConfig,
                                 logger=None):
    # ==================================================
    # Initialize environment
    environment = _initialize_environment(cfg)

    # ==================================================
    # Initialize agent
    agent = _initialize_agent(cfg, environment)

    # ==================================================
    # Pre-compute items like the true value
    pre_comput_dict = _pre_compute(cfg, environment, agent)

    # ==================================================
    # Run experiment
    for episode_idx in range(cfg.training.num_episodes):
        # Reset env
        obs = environment.reset()
        action = agent.begin_episode(obs)

        # Logging
        cumulative_reward = 0.0
        steps = 0

        while True:
            # Interact with environment
            obs, reward, done, info = environment.step(action)
            action = agent.step(obs, reward, done)

            # Tracker variables
            cumulative_reward += reward
            steps += 1

            if done:
                # ==
                # Log
                post_epis_dict = {
                    'episode_idx': episode_idx,
                    'total_steps': steps,
                    'cumulative_reward': cumulative_reward,
                }
                write_post_episode_log(cfg=cfg,
                                       pre_comput_dict=pre_comput_dict,
                                       episode_dict=post_epis_dict,
                                       environment=environment,
                                       agent=agent,
                                       logger=logger)

                # ==
                # Terminate
                break


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))  # print the configs

    # =====================================================
    # Initialize logger
    dc_fields = [k for k in vars(LogTupStruct())]  # temp object to get name
    log_title_str = '||'.join(dc_fields)
    if cfg.logging.dir_path is not None:
        log_path = os.path.join(cfg.logging.dir_path, 'progress.csv')
        logger = init_logger(log_path)
        logger.info(log_title_str)
    else:
        print(log_title_str)
        logger = None

    # =====================================================
    # Initialize experiment
    run_single_linear_experiment(cfg=cfg, logger=logger)


if __name__ == "__main__":
    main()
