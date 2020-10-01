# =============================================================================
# Training the agent
#
# Author: Anthony G. Chen
# =============================================================================

import argparse
import random
from itertools import product

import gym
import numpy as np
import pandas as pd
from tqdm import tqdm

from algos.lambda_agent import LambdaAgent
from algos.exp_strace_agent import STraceAgent
from envs.random_chain import RandomChainEnv


def run_chain_env(agentCls, agent_kwargs, num_episodes=10, n_states=19, seed=0):
    """
    Run training for a single agent and environment for a number of episodes
    :param agentCls:
    :param agent_kwargs:
    :param num_episodes:
    :param n_states:
    :param seed:
    :return:
    """

    # =====================================================
    # Initialize environment
    environment = RandomChainEnv(n_states, seed=seed)

    # =====================================================
    # Initialize Agent
    agent_kwargs['n_states'] = n_states
    agent_kwargs['seed'] = seed
    agent = agentCls(**agent_kwargs)

    # =====
    # Logging items
    V_mat = np.empty((num_episodes, n_states))
    epis_len_list = np.empty(num_episodes)
    cumu_rew_list = np.empty(num_episodes)

    # =====
    # Run
    for episode_idx in range(num_episodes):
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
                # Save items
                V_mat[episode_idx, :] = agent.V
                epis_len_list[episode_idx] = steps
                cumu_rew_list[episode_idx] = cumulative_reward

                break

    # =====================================================
    # Process information and return

    rmse_vec = compute_rmse(V_mat, agent_kwargs['gamma'])

    run_info = {
        'V_mat': V_mat,
        'rmse_vec': rmse_vec,
        'episode_lengths': epis_len_list,
        'cumulative_rewards': epis_len_list
    }

    # if this is an strace agent, save matrix
    if hasattr(agent, 'sTrace'):
        run_info['sTrace_matrix'] = agent.sTrace

    return run_info


def compute_rmse(vMat, gamma):
    """
    Compute the root mean squared error (RMSE) for the random chain env
    NOTE: currently only works in the UNDISCOUNTED case
    :param vMat: (n, m) matrix of value function over training, where
                 n is num episodes, m is number states
    :param gamma: discount value, not used for now
    :return: (n,) vector denoting RMSE for each episode
    """

    n_chain = np.shape(vMat)[1]

    # ==
    # Construct the true undiscounted value
    # NOTE: assuming it follows the pattern TODO prove the pattern
    numerators = [float(n) for n in range(1, n_chain+1)]
    true_V = np.array(numerators) / (n_chain + 1)
    true_V = true_V.reshape((1, n_chain))

    # ==
    # Compute RMSE
    avg_sq_err = np.average(((vMat - true_V)**2), axis=1)
    root_mse = np.sqrt(avg_sq_err)

    return root_mse


def run_chain_experiments(agentCls, indep_vars):
    """
    Run random chain expeirments with variables
    :param agentCls: agent class
    :param indep_vars: dict of lists for indep variables
    :return: pandas.DataFrame for all runs
    """
    # ==
    # Logging dict
    df_dict = {
        'gamma': [],
        'lr': [],
        'lambda': [],
        'seed': [],
        'num_episodes': [],
        'n_states': [],
        'final_rmse': []
    }

    # ==
    # Run experiments

    # Make into nested list for cartesian prod, and compute total exps
    indep_vars_lists = [indep_vars[k] for k in indep_vars]
    total_n_exp = 1
    for ele in indep_vars_lists:
        total_n_exp *= len(ele)

    # Run
    cartesian_prod = product(*indep_vars_lists)
    for attri_tup in tqdm(cartesian_prod, total=total_n_exp):
        lr, lambd, cur_seed = attri_tup

        gamma = 1.0
        num_episodes = 10
        n_states = 19

        agent_kwargs = {
            'gamma': gamma,
            'lamb': lambd,
            'lr': lr,
        }

        random_chain_kwargs = {
            'agentCls': agentCls,
            'agent_kwargs': agent_kwargs,
            'num_episodes': num_episodes,
            'n_states': n_states,
            'seed': cur_seed
        }

        run_info = run_chain_env(**random_chain_kwargs)

        df_dict['gamma'].append(gamma)
        df_dict['lr'].append(lr)
        df_dict['lambda'].append(lambd)
        df_dict['seed'].append(cur_seed)
        df_dict['num_episodes'].append(num_episodes)
        df_dict['n_states'].append(n_states)
        df_dict['final_rmse'].append(run_info['rmse_vec'][-1])

    df_run = pd.DataFrame.from_dict(df_dict)
    return df_run


if __name__ == "__main__":
    indep_vars = {
        'lr_list': [0.001, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'lambd_list': [1.0, 0.99, 0.975, 0.95, 0.9, 0.8, 0.4, 0.0],
        'seed_list': [s * 2 for s in range(50)]
    }

    # agentCls = LambdaAgent
    agentCls = STraceAgent

    df_run = run_chain_experiments(agentCls, indep_vars)

    # Save dataframe
    df_out_dir = '/network/tmp1/chenant/ant/exp_foward_trace/09-30/exp1_lambda'
    df_out_path = f'{df_out_dir}/sTrace_1srinit_runs.csv'
    df_run.to_csv(df_out_path)
    print(f'DF saved to: {df_out_path}')





def old_main():
    # TODO: have a hyperparmeter .config file for the future

    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run env')

    parser.add_argument('--num-episode', type=int, default=20, metavar='N',
                        help='number of episodes to run the environment for (default: 500)')

    parser.add_argument('--discount-factor', type=float, default=0.99, metavar='g',
                        help='discount factor (gamma) for future reward (default: 0.99)')

    # Experimental parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-path', type=str, default=None,
                        help='file path to the log file (default: None, printout instead)')
    parser.add_argument('--tmpdir', type=str, default='./',
                        help='temporary directory to store dataset for training (default: cwd)')

    args = parser.parse_args()
    print(args)

    # =====================================================
    # Initialize GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # =====================================================
    # Initialize logging
    # TODO maybe do this
    # Maybe TODO see logging procedures at
    # https://github.com/im-ant/ElectrophysRL/blob/master/dopatorch/discrete_domains/train.py

    # =====================================================
    # Start environmental interactions
    # run_environment(args, device=device, logger=logger)
    run_chain_experiments()
