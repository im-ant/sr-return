# =============================================================================
# Training the linear agent on Boyan's chain
#
# Author: Anthony G. Chen
# =============================================================================

import argparse
import configparser
from collections import namedtuple
import logging
import os
from itertools import product

import gym
import numpy as np
from tqdm import tqdm

from algos.sf_return_ag import SFReturnAgent
from algos.td_lambda_ag import SarsaLambdaAgent
from envs.boyans_chain import BoyansChainEnv

# Things to log
LogTupStruct = namedtuple('Log', field_names=['num_episodes',  # experiment-specific
                                              'agentCls_name',
                                              'seed',
                                              'gamma',
                                              'lr',
                                              'lamb',
                                              'use_true_R_fn',
                                              'episode_idx',  # episode-specific logs
                                              'total_steps',
                                              'cumulative_reward',
                                              'v_fn_rmse',
                                              ])


def init_logger(logging_path: str) -> logging.Logger:
    """
    Initializes the path to write the log to
    :param logging_path: path to write the log to
    :return: logging.Logger object
    """
    logger = logging.getLogger('Experiment-Log')

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logging_path)
    formatter = logging.Formatter('%(asctime)s||%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def solve_boyan_value_fn():
    """
    Solve the value function for each state of the 13-state Boyan's chain
    :return:
    """
    # Transition matrix
    P_trans = np.zeros((14, 14))
    for i in reversed(range(3, 14)):
        P_trans[i, i - 1] = 0.5
        P_trans[i, i - 2] = 0.5
    P_trans[2, 1] = 1.0
    P_trans[1, 0] = 1.0

    # Reward function
    # NOTE: charactering each state's reward by the expected reward
    #       from transitioning out of the state. This may be slightly
    #       different from the TD way of solving for state values.
    R_fn = np.ones(14) * (-3.0)
    R_fn[2] = -2.0
    R_fn[1] = 0.0
    R_fn[0] = 0.0

    gamma = 1.0
    c_mat = (np.identity(14) - (gamma * P_trans))
    v = np.linalg.inv(c_mat) @ R_fn

    return v


def compute_rmse(vec_a, vec_b):
    sq_err = (vec_a - vec_b) ** 2
    return np.sqrt(np.mean(sq_err))


def compute_value_rmse(env, agent, true_v_fn):
    """
    Compute the RMSE for the value function of a given agent.
    Here hard-coded for the 14-state Boyan's chain.

    :return: scalar RMSE
    """
    n_states = 14

    esti_v_fn = np.empty(n_states)

    for s_n in range(n_states):
        # Get state features
        s_phi = env.state_2_features(s_n)

        # Compute the value estimate TODO change this
        # NOTE: assumes only a single action is available
        esti_v_fn[s_n] = agent.compute_Q_value(s_phi, 0)
        
    return compute_rmse(esti_v_fn, true_v_fn)


def solve_linear_R_params(env):
    """
    Solve the parameters of a linear reward function on the 13-state
    Boyan's chain

    :param env: the Boyan's chain environment
    :return: (feature_dim, ) parameters of best fit linear reward function
    """

    n_states = 13 + 1
    feature_dim = env.observation_space.shape[0]  # assume linear

    # Fill matrices
    phi_mat = np.empty((n_states, feature_dim))
    r_vec = np.empty(n_states)

    for s_n in range(n_states):
        # Get state features
        phi_mat[s_n] = env.state_2_features(s_n)

        # Get reward
        if s_n > 2:
            s_R = -3.0
        elif s_n == 2:
            s_R = -2.0
        else:
            s_R = 0.0
        r_vec[s_n] = s_R

    # Solve parameters from Moore–Penrose inverse
    phi_mat_T = np.transpose(phi_mat)
    mp_inv = np.linalg.inv((phi_mat_T @ phi_mat)) @ phi_mat_T
    bf_Wr = mp_inv @ r_vec

    return bf_Wr


def run_single_boyans_chain(exp_kwargs: dict,
                            args, logger=None):
    # ==================================================
    # Initialize environment
    environment = BoyansChainEnv(
        exp_kwargs['seed']
    )

    # ==================================================
    # Initialize agent
    agentCls = exp_kwargs['agentCls']
    agent_kwargs = {
        'gamma': exp_kwargs['gamma'],
        'lamb': exp_kwargs['lamb'],
        'lr': exp_kwargs['lr'],
        'seed': exp_kwargs['seed'],
    }
    agent = agentCls(
        feature_dim=environment.observation_space.shape[0],  # assume linear
        num_actions=environment.action_space.n,
        **agent_kwargs
    )

    # ==================================================
    # Pre-compute and pre-initialize

    # Dictionary to log the things that don't change over each episode
    exp_log_dict = {}
    for k in exp_kwargs:
        if 'Cls' in k:
            exp_log_dict[f'{k}_name'] = exp_kwargs[k].__name__
        else:
            exp_log_dict[k] = exp_kwargs[k]

    # Compute the true Boyan's chain value function
    true_v_fn = solve_boyan_value_fn()

    # (Optional) Give agent the best fit reward fn weights
    if exp_kwargs['use_true_R_fn']:
        bf_Wr = solve_linear_R_params(environment)
        agent.Wr = bf_Wr
        agent.use_true_R_fn = True

    # ==================================================
    # Run experiment
    for episode_idx in range(exp_kwargs['num_episodes']):
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
                # Compute value function RMSE
                v_fn_rmse = compute_value_rmse(environment, agent, true_v_fn)

                # ==
                # Construct logging items

                # Experiment specific logs
                epis_log_dict = {k: exp_log_dict[k] for k in exp_log_dict}

                # Episode specific logs
                epis_log_dict['episode_idx'] = episode_idx
                epis_log_dict['total_steps'] = steps
                epis_log_dict['cumulative_reward'] = cumulative_reward
                epis_log_dict['v_fn_rmse'] = v_fn_rmse

                # TODO: should log more things relating to the value fn, SF, loss etc.

                # Construct log output
                logtuple = LogTupStruct(**epis_log_dict)
                log_str = '||'.join([str(e) for e in logtuple])
                if logger is not None:
                    logger.info(log_str)
                else:
                    if (episode_idx + 1) % 100 == 0:  # NOTE TODO hacky modulo
                        print(log_str)
                # ==
                # Terminate
                break

    # Maybe: save the SR matrix?


def run_experiments(args, config: configparser.ConfigParser, logger=None):
    # ==================================================
    # Parse config file

    # Training attributes
    num_episodes = config['Training'].getint('num_episodes')

    # Get agent attributes
    agentCls = globals()[config['Agent']['cls_string']]  # NOTE super hacky
    use_true_R_fn = config['Agent'].getboolean('use_true_R_fn')

    # Get comma-sep list of attributes
    cs_gamma = config['Agent']['cs_gamma']
    cs_lamb = config['Agent']['cs_lamb']
    cs_lr = config['Agent']['cs_lr']
    cs_seed = config['Training']['cs_seed']

    # Get the list of attributes
    indep_vars_dict = {
        'agentCls': [agentCls],
        'num_episodes': [num_episodes],
        'use_true_R_fn': [use_true_R_fn],
        'gamma': [float(e.strip()) for e in cs_gamma.split(',')],
        'lamb': [float(e.strip()) for e in cs_lamb.split(',')],
        'lr': [float(e.strip()) for e in cs_lr.split(',')],
        'seed': [int(e.strip()) for e in cs_seed.split(',')],
    }

    # ==================================================
    # Construct experimental variables

    # Unpack list
    indep_vars_keys = []
    indep_vars_lists = []
    total_num_exps = 1  # counting num experiments
    for k in indep_vars_dict:
        indep_vars_keys.append(k)
        indep_vars_lists.append(indep_vars_dict[k])
        total_num_exps *= len(indep_vars_dict[k])

    # Take Cartesian product and make iterable
    cartesian_prod = product(*indep_vars_lists)

    # Set iterable verbosity
    if args.progress_verbosity == 'tqdm':
        attri_iterable = tqdm(cartesian_prod, total=total_num_exps)
    else:
        attri_iterable = cartesian_prod

    # ==================================================
    # Run experiments
    for attri_tup in attri_iterable:
        exp_kwargs = {indep_vars_keys[i]: attri_tup[i]
                      for i in range(len(attri_tup))}
        run_single_boyans_chain(exp_kwargs, args, logger=logger)


if __name__ == "__main__":
    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file
    parser.add_argument(
        '--config_path', type=str,
        default='/home/mila/c/chenant/repos/sr-return/linear/default_config.ini',
        help='path to the agent configuration .ini file'
    )

    parser.add_argument(
        '--log_dir', type=str,
        default=None,
        help='file path to the experimental log directory (default: None)'
    )

    parser.add_argument(
        '--progress_verbosity', type=str, default='0',
        help='How often to print progress of experiments ran '
             '[int (per x experiments, 0 for no print), tqdm for '
             'tqdm progress bar] (default: "0")')

    # Parse arguments
    args = parser.parse_args()
    print(args)

    # =====================================================
    # Parse configurations
    config = configparser.ConfigParser()
    config.read(args.config_path)

    # =====================================================
    # Initialize logger
    log_title_str = '||'.join(LogTupStruct._fields)
    if args.log_dir is not None:
        log_path = os.path.join(args.log_dir, 'progress.csv')
        logger = init_logger(log_path)
        logger.info(log_title_str)
    else:
        print(log_title_str)
        logger = None

    # =====================================================
    # Run experiments
    run_experiments(args, config=config, logger=logger)
