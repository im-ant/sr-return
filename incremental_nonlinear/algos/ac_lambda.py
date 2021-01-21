# ============================================================================
# Original Authors:
# Kenny Young (kjyoung@ualberta.ca)
# Tian Tian (ttian@ualberta.ca)
#
# References used for this implementation:
#   https://pytorch.org/docs/stable/nn.html#
#   https://pytorch.org/docs/stable/torch.html
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# ============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class ACLambda:
    """
    Incremental AC lambda agent
    """

    def __init__(self, ModelCls, model_kwargs,
                 discount_gamma=0.99,
                 lr_alpha=0.00048828125,
                 trace_lambda=0.8,
                 entropy_beta=0.01,
                 grad_rms_gamma=0.999,
                 grad_rms_eps=0.0001,
                 min_denom=0.0001
                 ):
        # TODO: input everything as argument

        self.ModelCls = ModelCls
        self.model_kwargs = model_kwargs

        self.discount_gamma = discount_gamma
        self.lr_alpha = lr_alpha
        self.trace_lambda = trace_lambda
        self.entropy_beta = entropy_beta
        self.grad_rms_gamma = grad_rms_gamma
        self.grad_rms_eps = grad_rms_eps
        self.min_denom = min_denom

        # Network for AC evaluation
        self.model = None
        # Eligibility self.traces are stored here
        self.traces = []
        # Space allocated to store gradients used in training
        self.grads = []
        # Running average of mean squared gradient for use in RMSProp
        self.msgrads = []

    def initialize(self, env, device):
        """Initialize agent"""
        in_channels = env.state_shape()[2]
        num_actions = env.num_actions()

        self.model = self.ModelCls(
            in_channels,
            num_actions,
            **self.model_kwargs,
        ).to(device)

        # List of parameter names to add eligibility traces to
        trace_param_list = [
            'conv.weight', 'conv.bias', 'fc_hidden.weight', 'fc_hidden.bias',
            'policy.weight', 'policy.bias',
            'value.weight', 'value.bias',  # 'sf_fn.weight', 'sf_fn_bias',
        ]

        for name, params in self.model.named_parameters():
            if name not in trace_param_list:
                continue
            self.traces.append(torch.zeros(
                params.size(), dtype=torch.float32, device=device)
            )
            self.grads.append(torch.zeros(
                params.size(), dtype=torch.float32, device=device)
            )
            self.msgrads.append(torch.zeros(
                params.size(), dtype=torch.float32, device=device)
            )

        print(self.model)  # TODO delete?

    def optimize_agent(self, sample, time_step):

        # states, next_states: (1, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal: (1, 1)
        last_state = sample.last_state
        state = sample.state
        action = sample.action
        reward = sample.reward
        is_terminal = sample.is_terminal

        pi, V_curr = self.model(state)

        # Compute the targets
        # NOTE: i think this is basically like a sum of the losses used to compute the
        #       gradients
        trace_potential = V_curr + 0.5 * torch.log(pi[0, action] + self.min_denom)
        entropy = -torch.sum(torch.log(pi + self.min_denom) * pi)

        # Save the value + policy loss gradient
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
                # TD error
                V_last = self.model(last_state)[1]
                delta = self.discount_gamma * (0 if is_terminal else V_curr) + reward - V_last

                # Update uses RMSProp with initialization debiasing
                for param, trace, ms_grad in zip(self.model.parameters(),
                                                 self.traces,
                                                 self.msgrads):
                    grad = trace * delta[0] + self.entropy_beta * param.grad
                    ms_grad.copy_(self.grad_rms_gamma * ms_grad + (1 - self.grad_rms_gamma) * grad * grad)
                    # Param updates
                    param.copy_(
                        param + self.lr_alpha * grad / (torch.sqrt(
                            ms_grad / (1 - self.grad_rms_gamma ** (time_step + 1)) + self.grad_rms_eps
                        ))
                    )

        # Accumulating trace (Always update trace)
        with torch.no_grad():
            for grad, trace in zip(self.grads, self.traces):
                trace.copy_(self.trace_lambda * self.discount_gamma * trace + grad)

        # ==
        # Construct dict
        out_dict = None
        # TODO log the average trace magnitude?
        if last_state is not None:
            out_dict = {
                'value_loss': delta.item() ** 2,
            }

        return out_dict

    def clear_eligibility_traces(self):
        """
        Called to clear the elig traces at the end of an episode
        :return: None
        """
        for trace in self.traces:
            trace.zero_()


if __name__ == '__main__':
    pass
