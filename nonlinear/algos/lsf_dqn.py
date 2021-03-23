# ============================================================================
# DQN implementation with lambda value function
#
# Anthony G. Chen
# ============================================================================
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from algos.dqn import DQN

# ==
# Logging related
LogTupFields = ['Q_loss', 'SF_loss', 'R_loss',
                'num_policy_updates', 'grad_norm']
LogTupStruct = namedtuple(
    'LogTupStruct',
    field_names=LogTupFields, defaults=(None, ) * len(LogTupFields)
)


# ==
# Agent algorithms

class LSF_DQN(DQN):
    """
    Deep Q network with lambda value function
    """

    def __init__(self, ModelCls, model_kwargs,
                 discount_gamma=0.99,
                 sf_lambda=0.0,
                 start_epsilon=1.0,
                 end_epsilon=0.1,
                 initial_epsilon_length=5000,
                 epsilon_anneal_length=100000,
                 use_target_net=True,
                 policy_updates_per_target_update=1000,
                 optim_kwargs=None,
                 seed=None,
                 ):
        super().__init__(
            ModelCls, model_kwargs, discount_gamma=discount_gamma,
            start_epsilon=start_epsilon, end_epsilon=end_epsilon,
            initial_epsilon_length=initial_epsilon_length,
            epsilon_anneal_length=epsilon_anneal_length,
            use_target_net=use_target_net,
            policy_updates_per_target_update=policy_updates_per_target_update,
            optim_kwargs=optim_kwargs,
            seed=seed,
        )

        # Model class and parameters
        self.ModelCls = ModelCls
        self.model_kwargs = model_kwargs
        self.num_actions = None

        self.discount_gamma = discount_gamma
        self.sf_lambda = sf_lambda

        # Actor parameters  # TODO update / override this?
        self.actor_kwargs = {
            'start_epsilon': start_epsilon,
            'end_epsilon': end_epsilon,
            'initial_epsilon_length': initial_epsilon_length,
            'epsilon_anneal_length': epsilon_anneal_length,
            'sf_lambda': self.sf_lambda,
            'seed': seed
        }
        self.actor = None

        # Optimization parameters
        self.optim_kwargs = optim_kwargs
        self.optimizer = None

        self.policy_updates_per_target_update = policy_updates_per_target_update

        self.rng = np.random.default_rng(seed)

        # Logging
        self.logTupStruct = LogTupStruct

        # Counter variables
        self.policy_updates_counter = 0

    def initialize(self, env, device):
        """Initialize agent at train-time"""
        in_channels = env.state_shape()[2]
        num_actions = env.num_actions()
        self.num_actions = num_actions

        # Instantiate networks
        self.policy_net = self.ModelCls(
            in_channels, num_actions,
            **self.model_kwargs).to(device)
        self.model = self.policy_net  # TODO: this needs to be by reference

        if self.use_target_net:
            self.target_net = self.ModelCls(
                in_channels, num_actions,
                **self.model_kwargs).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.target_net = self.policy_net

        # Instantiate optimizer
        if self.optim_kwargs is None:
            self.optim_kwargs = {
                'lr': 0.00025,
                'alpha': 0.95,
                'centered': True,
                'eps': 0.01,
            }  # TODO: confirm with MinAtar code to see if correspond
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            **self.optim_kwargs,
        )

        # Init actor
        actor_kwargs = {
            'model': self.model,
            'num_actions': self.num_actions,
            'device': device,
            **self.actor_kwargs,
        }
        self.actor = LSF_DQNActor(**actor_kwargs)

        print('self.optim_kwargs', self.optim_kwargs)
        print('actor_kwargs', actor_kwargs)
        print(self.policy_net)  # maybe todo delete these?

    def episode_reset(self):
        pass

    def get_action(self, state, total_steps):
        """
        Get action with epsilon greedy policy
        """
        return self.actor.get_action(state, total_steps)

    def compute_loss(self, sample):
        # ==
        # Unpack batched sample
        states = sample.state  # (batch_n, channel, height, width)
        next_states = sample.next_state  # (batch_n, c, h, w)
        actions = sample.action  # (batch_n, 1)
        rewards = sample.reward  # (batch_n, 1)
        is_terminal = sample.is_terminal  # (batch_n, 1)

        # ==
        # Compute current estimates
        cur_out_tup = self.policy_net.compute_estimates(states, actions,
                                                        self.sf_lambda)
        cur_phi, SF_s_a, Q_s_a, R_s, __ = cur_out_tup

        # ==
        # Next state estimates
        with torch.no_grad():
            nex_out_tup = self.target_net.compute_targets(
                next_states, self.sf_lambda)
            nex_phi, nex_maxQ_sf, nex_maxQ_val = nex_out_tup

            SF_sp_ap = nex_maxQ_sf * (~is_terminal)  # (batch_n, d)
            Q_sp_ap = nex_maxQ_val * (~is_terminal)  # (batch_n, 1)

            # Targets
            SF_target = cur_phi.detach() + (
                (self.discount_gamma * self.sf_lambda) * SF_sp_ap.detach()
            )  # (batch_n, d)
            Q_target = rewards + (
                    self.discount_gamma * Q_sp_ap.detach()
            )  # (batch_n, 1)

        # ==
        # Losses
        SF_loss = f.smooth_l1_loss(SF_target, SF_s_a)
        Q_loss = f.smooth_l1_loss(Q_target, Q_s_a)
        R_loss = f.smooth_l1_loss(rewards, R_s)

        loss = Q_loss + SF_loss + R_loss
        info = {
            'Q_loss': Q_loss.item(),
            'SF_loss': SF_loss.item(),
            'R_loss': R_loss.item(),
        }

        return loss, info

    def post_update_step(self, loss):
        # ==
        # Update target network
        if (self.use_target_net
                and (self.policy_updates_counter > 0)
                and (self.policy_updates_counter
                     % self.policy_updates_per_target_update == 0)):
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        # ==
        # Counters and output
        self.policy_updates_counter += 1

        total_grad_norm = 0.0
        total_param_count = 0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
            total_param_count += 1

        out_dict = {
            'num_policy_updates': self.policy_updates_counter,
            'grad_norm': (total_grad_norm/total_param_count),
        }
        # TODO add gradient norm, epsilon, etc.

        return out_dict

    def optimize_agent(self, sample, total_steps):
        # Compute loss
        loss, info = self.compute_loss(sample)

        # Zero gradients, backprop, update the weights of policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update
        out_dict = self.post_update_step(loss)
        return {**info, **out_dict}


class LSF_DQNActor:
    def __init__(self, model, num_actions,
                 start_epsilon, end_epsilon,
                 initial_epsilon_length, epsilon_anneal_length,
                 sf_lambda,
                 device,
                 seed=0):
        self.model = model  # policy network
        self.device = device

        self.num_actions = num_actions
        self.sf_lambda = sf_lambda

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.initial_epsilon_length = initial_epsilon_length
        self.epsilon_anneal_length = epsilon_anneal_length

        self.rng = np.random.default_rng(seed)

    def set_model(self, model):
        self.model = model

    def get_action(self, state, total_steps):
        """
        Get action with epsilon greedy policy
        :param state:
        :param total_steps:
        :return:
        """
        if total_steps < self.initial_epsilon_length:
            # Initial uniform random action
            action = torch.tensor([[self.rng.integers(self.num_actions)]],
                                  device=self.device)
        else:
            # Compute current epsilon for random policy
            if ((total_steps - self.initial_epsilon_length)
                    >= self.epsilon_anneal_length):
                # Post-annealing epsilon
                eps = self.end_epsilon
            else:
                # During-annealing epsilon
                eps_stepsize = ((self.end_epsilon - self.start_epsilon)
                                / self.epsilon_anneal_length)
                eps_steps = total_steps - self.initial_epsilon_length
                eps = eps_stepsize * eps_steps + self.start_epsilon

            if self.rng.binomial(1, eps) == 1:
                # If epsilon uniform random action
                action = torch.tensor([[self.rng.integers(self.num_actions)]],
                                      device=self.device)
            else:
                # Greedy action
                # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
                # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
                # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
                with torch.no_grad():
                    action = self.model(state, self.sf_lambda).max(1)[1].view(1, 1)
                    # TODO: confirm this will always be on self.device

        return action


if __name__ == '__main__':
    pass
