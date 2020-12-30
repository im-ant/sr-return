# =============================================================================
# Utility functions for solving MDPs
#
# Author: Anthony G. Chen
# =============================================================================

import gym
import numpy as np


def compute_rmse(vec_a, vec_b) -> float:
    """
    Compute the root mean square error (RMSE) between two vectors
    :param vec_a: np array of shape (N, )
    :param vec_b: np array of shape (N, )
    :return: scalar
    """
    sq_err = (vec_a - vec_b) ** 2
    return np.sqrt(np.mean(sq_err))


def solve_value_fn(env: gym.Env, gamma: float) -> np.ndarray:
    """
    Solve the value functio for each state in an environment
    :param env: gym environment
    :param gamma: discount factor
    :return:
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


def solve_linear_sf(env: gym.Env, discount_factor: float) -> np.ndarray:
    """
    Solve for the linear successor features given an environment
    :param env: gym environment
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


def compute_value_rmse(env: gym.Env, agent, true_v_fn) -> float:
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


def compute_sf_ret_rmse(env, agent, true_v_fn) -> float:
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


if __name__ == "__main__":
    print('hello world')
