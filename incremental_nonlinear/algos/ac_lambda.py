##############################################################################
# Original Authors:
# Kenny Young (kjyoung@ualberta.ca)
# Tian Tian (ttian@ualberta.ca)
#
# References used for this implementation:
#   https://pytorch.org/docs/stable/nn.html#
#   https://pytorch.org/docs/stable/torch.html
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
##############################################################################


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


#####################################################################################################################
# train
#
# This is where learning happens. More specifically, this function updates the weights of the policy/value self.model.
#
# Inputs:
#   sample: a single transition
#   self.traces: an instance of Qself.model
#   grads: a list of tensors, one for each self.model parameter. Used as temporary storage of computed gradient
#   self.model: an instance of ACself.model, to be trained
#   alpha: learning rate for actor-critic update
#
#####################################################################################################################

class ACLambda:
    """
    Incremental AC lambda agent
    """

    def __init__(self, ModelCls, model_kwargs):
        # TODO: input everything as argument

        self.ModelCls = ModelCls
        self.model_kwargs = model_kwargs

        self.model = None
        # Eligibility self.traces are stored here
        self.traces = None
        # Space allocated to store gradients used in training
        self.grads = None
        # Running average of mean squared gradient for use in RMSProp
        self.msgrads = None

        self.lr_alpha = 0.00048828125
        self.trace_lambda = 0.8
        self.discount_gamma = 0.99
        self.grad_beta = 0.01
        self.grad_rms_gamma = 0.999
        self.grad_rms_eps = 0.0001
        self.min_denom = 0.0001

    def initialize(self, env, device):
        """Initialize agent"""
        in_channels = env.state_shape()[2]
        num_actions = env.num_actions()

        self.model = self.ModelCls(
            in_channels,
            num_actions,
            **self.model_kwargs
        ).to(device)

        self.traces = [
            torch.zeros(x.size(),
                        dtype=torch.float32,
                        device=device) for x in self.model.parameters()
        ]
        self.grads = [
            torch.zeros(x.size(),
                        dtype=torch.float32,
                        device=device) for x in self.model.parameters()
        ]
        self.msgrads = [
            torch.zeros(x.size(),
                        dtype=torch.float32,
                        device=device) for x in self.model.parameters()
        ]

    def optimize_agent(self, sample, time_step):  # TODO change to optimize agent

        """
        LAMBDA = 0.8
        GAMMA = 0.99
        BETA = 0.01
        self.grad_rms_gamma = 0.999
        EPS_RMS = 0.0001
        MIN_DENOM = 0.0001
        """

        # states, next_states: (1, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal: (1, 1)
        last_state = sample.last_state
        state = sample.state
        action = sample.action
        reward = sample.reward
        is_terminal = sample.is_terminal

        pi, V_curr = self.model(state)

        # Compute the targets
        trace_potential = V_curr + 0.5 * torch.log(pi[0, action] + self.min_denom)
        entropy = -torch.sum(torch.log(pi + self.min_denom) * pi)

        self.model.zero_grad()
        trace_potential.backward(retain_graph=True)

        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), self.grads):
                grad.data.copy_(param.grad)

        # Update parameters except for on the first observation
        if last_state is not None:
            self.model.zero_grad()
            entropy.backward()
            with torch.no_grad():
                V_last = self.model(last_state)[1]
                delta = self.discount_gamma * (0 if is_terminal else V_curr) + reward - V_last

                # Update uses RMSProp with initialization debiasing
                for param, trace, ms_grad in zip(self.model.parameters(),
                                                 self.traces,
                                                 self.msgrads):
                    grad = trace * delta[0] + self.grad_beta * param.grad
                    ms_grad.copy_(self.grad_rms_gamma * ms_grad + (1 - self.grad_rms_gamma) * grad * grad)
                    param.copy_(param + self.lr_alpha * grad / (
                        torch.sqrt(ms_grad / (1 - self.grad_rms_gamma ** (time_step + 1)) + self.grad_rms_eps)))

        # Always update trace
        with torch.no_grad():
            for grad, trace in zip(self.grads, self.traces):
                trace.copy_(self.trace_lambda * self.discount_gamma * trace + grad)

    def clear_eligibility_traces(self):
        """
        Called to clear the elig traces at the end of an episode
        :return: None
        """
        for trace in self.traces:
            trace.zero_()


if __name__ == '__main__':
    pass
