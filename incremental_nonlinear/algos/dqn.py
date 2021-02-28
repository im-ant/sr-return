# ============================================================================
# Modified from original MinAtar examples from authors:
# Kenny Young (kjyoung@ualberta.ca)
# Tian Tian (ttian@ualberta.ca)
#
# Anthony G. Chen
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class DQN:
    """
    Deep Q network training algorithm
    """

    def __init__(self, ModelCls, model_kwargs,
                 discount_gamma=0.99,
                 start_epsilon=1.0,
                 end_epsilon=0.1,
                 initial_epsilon_length=5000,
                 epsilon_anneal_length=100000,
                 use_target_net=True,
                 policy_updates_per_target_update=1000,
                 optim_kwargs=None,
                 seed=None,
                 ):

        # Model class and parameters
        self.ModelCls = ModelCls
        self.model_kwargs = model_kwargs
        self.num_actions = None

        self.discount_gamma = discount_gamma

        # Actor parameters
        self.actor_kwargs = {
            'start_epsilon': start_epsilon,
            'end_epsilon': end_epsilon,
            'initial_epsilon_length': initial_epsilon_length,
            'epsilon_anneal_length': epsilon_anneal_length,
            'seed': seed
        }
        self.actor = None

        # Models instances and references
        self.use_target_net = use_target_net
        self.policy_net = None
        self.target_net = None
        self.model = None

        # Optimization parameters
        self.optim_kwargs = optim_kwargs
        self.optimizer = None

        self.policy_updates_per_target_update = policy_updates_per_target_update

        self.rng = np.random.default_rng(seed)

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
        self.actor = DQNActor(**actor_kwargs)

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
        # Get Q(s_t,a_t) estimates in forward pass
        # by estimating all Q and only using the action taken
        Q_s_a = self.policy_net(states).gather(1, actions)  # (batch_n, 1)

        # ==
        # Get next step prediction
        # Note: tensor.max(1): get max values and indeces over cols
        #       tensor.max(1)[0].unsqueeze(1): max vals and restore orig dim
        Q_sp_ap = (
                self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
                * (~is_terminal)  # set to zero if is_terminal == True
        )  # (batch_n, 1)

        # Compute the target
        target = rewards + (self.discount_gamma * Q_sp_ap)

        # ==
        # Loss (Huber)
        loss = f.smooth_l1_loss(target, Q_s_a)
        return loss

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

        out_dict = {
            'value_loss': loss.item(),
        }
        # TODO add gradient norm, epsilon, etc.

        return out_dict

    def optimize_agent(self, sample, total_steps):
        # Compute loss
        loss = self.compute_loss(sample)

        # Zero gradients, backprop, update the weights of policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update
        return self.post_update_step(loss)


class DQNActor:
    def __init__(self, model, num_actions,
                 start_epsilon, end_epsilon,
                 initial_epsilon_length, epsilon_anneal_length,
                 device,
                 seed=0):
        self.model = model  # policy network
        self.device = device

        self.num_actions = num_actions

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
                    action = self.model(state).max(1)[1].view(1, 1)
                    # TODO: confirm this will always be on self.device

        return action


if __name__ == '__main__':
    pass
