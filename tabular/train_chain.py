# =============================================================================
# Training the agent
#
# Author: Anthony G. Chen
# =============================================================================

import argparse
from collections import namedtuple
import logging
import os
from itertools import product

import gym
import numpy as np
from tqdm import tqdm

from algos.lambda_agent import LambdaAgent
from algos.exp_strace_agent import STraceAgent
from algos.sr_agent import SRAgent
from envs.random_chain import RandomChainEnv

# Things to log
LogTupStruct = namedtuple('Log', field_names=['num_episodes',
                                              'n_states',
                                              'agentCls_name',
                                              'seed',
                                              'gamma',
                                              'lr',
                                              'lamb',
                                              's_subsample_prop',
                                              'use_true_s_mat',
                                              'use_rand_s_mat',
                                              'use_true_r_fn',
                                              'episode_idx',  # episode-specific logs
                                              'total_steps',
                                              'cumulative_reward',
                                              'v_vec_max',
                                              'v_vec_min',
                                              'v_vec_avg',
                                              'v_vec_rmse',
                                              's_mat_norm',
                                              's_mat_rmse',
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


def run_chain_env(exp_kwargs: dict, args, logger=None):
    """

    :param exp_kwargs:
    :param logger:
    :return:
    """

    # =====================================================
    # Initialize environment
    environment = RandomChainEnv(n_states=exp_kwargs['n_states'],
                                 seed=exp_kwargs['seed'])

    # =====================================================
    # Initialize Agent
    agentCls = exp_kwargs['agentCls']
    agent_kwargs = {
        'n_states': exp_kwargs['n_states'],
        'gamma': exp_kwargs['gamma'],
        'lamb': exp_kwargs['lamb'],
        'lr': exp_kwargs['lr'],
        's_prop_sample': exp_kwargs['s_subsample_prop'],
        'use_true_s_mat': exp_kwargs['use_true_s_mat'],
        'use_rand_s_mat': exp_kwargs['use_rand_s_mat'],
        'use_true_r_fn': exp_kwargs['use_true_r_fn'],
        'seed': exp_kwargs['seed'],
    }

    agent = agentCls(**agent_kwargs)

    # =====
    # Pre-initialize some logging items

    # Dictionary to log the things that don't change over each episode
    exp_log_dict = {}
    for k in exp_kwargs:
        if 'Cls' in k:
            exp_log_dict[f'{k}_name'] = exp_kwargs[k].__name__
        else:
            exp_log_dict[k] = exp_kwargs[k]

    # =====
    # Solve the MDP to compute against true values
    env_n_states = exp_kwargs['n_states']
    chain_T_mat = get_chain_MRP_transition(env_n_states)

    # Solve for lambda SR
    if hasattr(agent, 'S_mat'):
        agent_discount = agent.gamma * agent.lamb
        lamb_SR_mat = solve_SR(chain_T_mat, agent_discount)
    else:
        lamb_SR_mat = None

    # Solve for SR
    env_SR_mat = solve_SR(chain_T_mat, exp_kwargs['gamma'])

    # Compute value function
    true_V_fn = get_chain_value_fn(env_n_states, exp_kwargs['gamma'],
                                   sr_mat=env_SR_mat)

    # =====
    # Run episodes
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
                # Compute items
                v_vec_rmse = compute_rmse(true_V_fn, agent.V)
                s_mat_rmse = None
                if hasattr(agent, 'S_mat'):
                    s_mat_rmse = compute_rmse(lamb_SR_mat, agent.S_mat)
                    # TODO compute s mat norm?

                # ==
                # Construct logging items
                epis_log_dict = {k: exp_log_dict[k] for k in exp_log_dict}

                epis_log_dict['episode_idx'] = episode_idx
                epis_log_dict['total_steps'] = steps
                epis_log_dict['cumulative_reward'] = cumulative_reward
                epis_log_dict['v_vec_max'] = max(agent.V)
                epis_log_dict['v_vec_min'] = min(agent.V)
                epis_log_dict['v_vec_avg'] = np.mean(agent.V)
                epis_log_dict['v_vec_rmse'] = v_vec_rmse
                epis_log_dict['s_mat_norm'] = None
                epis_log_dict['s_mat_rmse'] = s_mat_rmse

                logtuple = LogTupStruct(**epis_log_dict)
                log_str = '||'.join([str(e) for e in logtuple])
                if logger is not None:
                    logger.info(log_str)
                else:
                    if (episode_idx + 1) % 10 == 0:
                        print(log_str)

                last_logtuple = logtuple

                # ==
                # Terminate
                break

    # ==
    # Save the agent S(R)-matrix
    if args.log_dir is not None:
        if hasattr(agent, 'S_mat'):
            # Contruct the file name and path
            attri_list = [f'{k}-{exp_log_dict[k]}' for k in exp_log_dict]
            attri_str = '__'.join([str(e) for e in attri_list])
            attri_str = attri_str.replace('.', 'd')
            s_mat_file_name = f'sMat__{attri_str}.npy'

            s_mat_file_path = os.path.join(args.log_dir, s_mat_file_name)

            # Write file
            np.save(s_mat_file_path, agent.S_mat)


def get_chain_MRP_transition(n_states: int):
    """
    Construct the transition matrix of a chain MDP with random transitions
    (i.e. Markov reward process)
    :param num_states: number of states in the chain
    :return: (n_states, n_states) matrix of transition
    """
    P_mat = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        P_mat[i, i + 1] = 0.5
        P_mat[i + 1, i] = 0.5
    P_mat[0, 0] = 0.0  # or 0.5?
    P_mat[-1, -1] = 0.0

    return P_mat


def get_chain_value_fn(n_states: int, discount_factor: float,
                       sr_mat=None):
    """
    Solve for the true value function of the MRP
    :param n_states:
    :param discount_factor:
    :param sr_mat:
    :return: (n_states,) vector, value fn
    """
    # Get future occupancy
    if sr_mat is None:
        T_mat = get_chain_MRP_transition(n_states)
        sr_mat = solve_SR(T_mat, discount_factor)

    # Generate reward fn
    R_fn = np.zeros(n_states)
    R_fn[-1] = 0.5  # TODO check correctness

    # Get value fn
    v_fn = np.dot(sr_mat, R_fn)
    return v_fn


def solve_SR(P_mat, discount_factor):
    """
    Solve for the future occupancy given a MRP transition matrix
    and discount factor
    :param P_mat: MRP transition matrix
    :param discount_factor: discount
    :return: (n_states, n_states) matrix of future disc occu
    """

    n_states = np.shape(P_mat)[0]

    # ==
    # Safeguard against singular matrix ?
    # df = min(discount_factor, 0.99999)
    df = discount_factor

    # ==
    # Solve the discounted occupancy problem
    c_mat = np.identity(n_states) - (df * P_mat)
    sr_mat = np.linalg.inv(c_mat)

    return sr_mat


def compute_rmse(target, prediction):
    """
    Compute the root mean squared error (RMSE) between two quantities
    Treat both as flattened vectors
    :param target:
    :param prediction:
    :return: scalar
    """

    # Always flatten
    t_vec = target.flatten()
    p_vec = prediction.flatten()

    # Compute RMSE
    sqr_err = (t_vec - p_vec) ** 2
    rmse = np.sqrt(np.average(sqr_err))
    return rmse


def run_chain_experiments(indep_vars_dict, args, logger=None):
    """
    Run random chain experiment with cartesian prod of indep vars
    :param indep_vars_dict:
    :param use_tqdm:
    :param logger:
    :return:
    """

    # ==
    # Unpack independent variables and take Cartesian product

    # Unpack list
    indep_vars_keys = []
    indep_vars_lists = []
    for k in indep_vars_dict:
        indep_vars_keys.append(k)
        indep_vars_lists.append(indep_vars_dict[k])

    # Count total number of experiments
    total_n_exp = 1
    for ele in indep_vars_lists:
        total_n_exp *= len(ele)

    # Take Cartesian product and make iterable
    cartesian_prod = product(*indep_vars_lists)

    # Construct iterable items
    if args.progress_verbosity == 'tqdm':
        attri_iterable = tqdm(cartesian_prod, total=total_n_exp)
    else:
        attri_iterable = cartesian_prod
    print_interval = None
    try:
        print_interval = int(args.progress_verbosity)
    except ValueError:
        pass

    # ==
    # Run experiments
    counter = 0
    for attri_tup in attri_iterable:
        counter += 1
        if (print_interval is not None
                and print_interval > 0
                and counter % print_interval == 0):
            print(f'Progress: [{counter}/{total_n_exp}]')

        exp_kwargs = {indep_vars_keys[i]: attri_tup[i]
                      for i in range(len(attri_tup))}

        run_chain_env(exp_kwargs, args, logger=logger)


if __name__ == "__main__":
    # =====================================================
    # Manual configs?
    # Using the same name as the logtuple keys

    # Agents: [LambdaAgent, STraceAgent, SRAgent]
    agentCls = STraceAgent

    indep_vars_dict = {
        'lr': [0.005, 0.01, 0.1],
        'lamb': [1.0],
        'seed': [s * 2 for s in range(50)],
        'gamma': [0.8],
        's_subsample_prop': [0.05],
        'use_true_s_mat': [False],
        'use_rand_s_mat': [False],
        'use_true_r_fn': [True],
        'num_episodes': [5000],
        'n_states': [19],
        'agentCls': [STraceAgent, SRAgent],
    }

    # indep_vars_dict['lr'] = np.array(indep_vars_dict['lr']) / 20.0


    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Random chain env')
    parser.add_argument('--progress_verbosity', type=str, default='0',
                        help='How often to print progress of experiments ran '
                             '[int (per x experiments, 0 for no print),'
                             ' tqdm for tqdm progress bar]'
                             '(default: "0")')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='file path to the experiment log directory'
                             '(default: None)')

    args = parser.parse_args()
    print(args)

    # =====================================================
    # Initialize logging
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
    run_chain_experiments(indep_vars_dict=indep_vars_dict,
                          args=args,
                          logger=logger)


def old_ref():
    """
    Keeping references

    ----------
    Sweep experiments
    ----------
    indep_vars_dict = {
        'lr': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08,
               0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'lamb': [1.0, 0.99, 0.975, 0.95, 0.9, 0.8, 0.5, 0.3, 0.0],
        'seed': [s * 2 for s in range(50)],
        'gamma': [1.0],
        's_subsample_prop': [0.05],
        'use_true_s_mat': [False],
        'use_rand_s_mat': [False],
        'use_true_r_fn': [False],
        'num_episodes': [10],
        'n_states': [19],
        'agentCls': [agentCls],
    }

    ----------
    Casual runs
    ----------
    indep_vars_dict = {
        'lr': [0.2],
        'lamb': [0.4],
        'seed': [s * 2 for s in range(10)],
        'gamma': [1.0],
        's_subsample_prop': [0.05],
        'use_true_s_mat': [True],
        'use_rand_s_mat': [False],
        'use_true_r_fn': [False],
        'num_episodes': [10],
        'n_states': [19],
        'agentCls': [agentCls],
    }


    """

    pass
