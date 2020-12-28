# =============================================================================
# Training the linear agent on linear prediction environments
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
from algos.expt_trace_ag import ExpectedTraceAgent
from envs.boyans_chain import BoyansChainEnv
from envs.random_walk_chain import RandomWalkChainEnv
from envs.perf_bin_tree import PerfBinaryTreeEnv

# Things to log
LogTupStruct = namedtuple('Log', field_names=['num_episodes',  # experiment-specific
                                              'envCls_name',
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
                                              'sf_G_rmse',
                                              'value_loss_avg',  # agent log dict specific logs
                                              'reward_loss_avg',
                                              'sf_loss_avg',
                                              'et_loss_avg',
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


def compute_rmse(vec_a, vec_b):
    sq_err = (vec_a - vec_b) ** 2
    return np.sqrt(np.mean(sq_err))


def solve_value_fn(env, gamma):
    """
    Solve the (tabular) value function for each state
    :return: tabular reward function, shaped (n_states, )
    """
    # Transition matrix
    n_states = env.get_num_states()
    P_trans = env.get_transition_matrix()

    # Reward function
    R_fn = env.get_reward_function()

    # Solve and return
    c_mat = (np.identity(n_states) - (gamma * P_trans))
    v_fn = np.linalg.inv(c_mat) @ R_fn

    return v_fn


def solve_linear_sf(env, discount_factor):
    """
    Solve for the linear successor features given an environment
    :param discount_factor: (gamma * lamb)
    :return:
    """
    phiMat = env.get_feature_matrix()
    transMat = env.get_transition_matrix()
    p_n_states = np.shape(transMat)[0]

    cMat = np.identity(p_n_states) - (discount_factor * transMat)
    qMat = cMat @ phiMat

    # Solve
    Z = np.linalg.inv(qMat.T @ qMat) @ qMat.T @ transMat @ phiMat
    return Z


def compute_value_rmse(env, agent, true_v_fn):
    """
    Compute the RMSE for the value function of a given agent
    and environment

    :return: scalar RMSE
    """
    n_states = env.get_num_states()

    esti_v_fn = np.empty(n_states)

    for s_n in range(n_states):
        # Get state features
        s_phi = env.state_2_features(s_n)

        # Compute the value estimate TODO change this
        # NOTE: assumes only a single action is available
        # TODO: make it into compute V function to marginalize over actions?
        esti_v_fn[s_n] = agent.compute_Q_value(s_phi, 0)

    return compute_rmse(esti_v_fn, true_v_fn)


def compute_sf_ret_rmse(env, agent, true_v_fn):
    """
    Compute RMSE for the lambda successor return, if possible
    :return: scalar RMSE
    """
    if not hasattr(agent, 'compute_successor_return'):
        return None

    n_states = env.get_num_states()
    esti_v_fn = np.empty(n_states)
    for s_n in range(n_states):
        s_phi = env.state_2_features(s_n)  # state features
        esti_v_fn[s_n] = agent.compute_successor_return(
            s_phi, 0
        )  # compute value
    return compute_rmse(esti_v_fn, true_v_fn)


def helper_extract_agent_log_dict(agent):
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


def run_single_lienar_experiment(exp_kwargs: dict,
                                 args, logger=None):
    # ==================================================
    # Initialize environment
    envCls = exp_kwargs['envCls']
    env_kwargs = {'seed': exp_kwargs['seed']}
    environment = envCls(**env_kwargs)

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
    true_v_fn = solve_value_fn(environment, exp_kwargs['gamma'])

    # (Optional) Give agent the best fit reward fn weights
    if exp_kwargs['use_true_R_fn']:
        bf_Wr = environment.solve_linear_reward_parameters()
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
                # Construct log item
                epis_log_dict = {k: exp_log_dict[k] for k in exp_log_dict}

                # ==
                # Compute Value function RMSE
                v_fn_rmse = compute_value_rmse(environment, agent, true_v_fn)
                sf_G_rmse = compute_sf_ret_rmse(environment, agent, true_v_fn)

                # ==
                # Episode log items

                # Episode specific logs
                epis_log_dict['episode_idx'] = episode_idx
                epis_log_dict['total_steps'] = steps
                epis_log_dict['cumulative_reward'] = cumulative_reward
                epis_log_dict['v_fn_rmse'] = v_fn_rmse
                epis_log_dict['sf_G_rmse'] = sf_G_rmse

                # ==
                # Compute losses from the agent logs
                agent_log_dict = helper_extract_agent_log_dict(agent)
                for k in agent_log_dict:
                    epis_log_dict[k] = agent_log_dict[k]

                # ==
                # Construct log output
                logtuple = LogTupStruct(**epis_log_dict)
                log_str = '||'.join([str(e) for e in logtuple])
                if logger is not None:
                    logger.info(log_str)
                else:
                    if (episode_idx + 1) % args.log_print_freq == 0:
                        print(log_str)

                #print(agent.Wq)  # TODO delete
                #print(obs)

                # ==
                # Terminate
                break

    # np.set_printoptions(precision=3)  # TODO delete below
    # print(np.shape(agent.Ws))
    # print(agent.Ws[0, 7:12, 7:12])

    # Maybe: save the SR matrix?


def run_experiments(args, config: configparser.ConfigParser, logger=None):
    # ==================================================
    # Parse config file

    # Training attributes
    num_episodes = config['Training'].getint('num_episodes')

    # Get environment attributes
    cs_envCls = config['Env']['cls_string']
    envCls_strs = [str(e.strip()) for e in cs_envCls.split(',')]
    envCls_list = [globals()[e] for e in envCls_strs]  # NOTE super hacky

    # Get agent attributes
    cs_agentCls = config['Agent']['cls_string']
    agentCls_strs = [str(e.strip()) for e in cs_agentCls.split(',')]
    agentCls_list = [globals()[e] for e in agentCls_strs]  # NOTE super hacky
    use_true_R_fn = config['Agent'].getboolean('use_true_R_fn')

    # Get comma-sep list of attributes
    cs_gamma = config['Agent']['cs_gamma']
    cs_lamb = config['Agent']['cs_lamb']
    cs_lr = config['Agent']['cs_lr']
    cs_seed = config['Training']['cs_seed']

    # Get the list of attributes
    indep_vars_dict = {
        'envCls': envCls_list,
        'agentCls': agentCls_list,
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
        run_single_lienar_experiment(exp_kwargs, args, logger=logger)


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

    parser.add_argument(
        '--log_print_freq', type=int, default=100,
        help='Frequency (number of episodes) to print the log; only prints '
             'when there is no log file to write to (default: 100)')

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
