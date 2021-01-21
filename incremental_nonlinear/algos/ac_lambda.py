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
        self.traces = None
        # Space allocated to store gradients used in training
        self.grads = None
        # Running average of mean squared gradient for use in RMSProp
        self.msgrads = None

        # List of allowable parameters to include for eligibility traces
        self.trace_param_list = [
            'conv.weight', 'conv.bias', 'fc_hidden.weight', 'fc_hidden.bias',
            'policy.weight', 'policy.bias',
            'value.weight', 'value.bias',
        ]

    def initialize(self, env, device):
        """Initialize agent"""
        in_channels = env.state_shape()[2]
        num_actions = env.num_actions()

        self.model = self.ModelCls(
            in_channels,
            num_actions,
            **self.model_kwargs,
        ).to(device)

        self.traces = {}
        self.grads = {}
        self.msgrads = {}

        for name, param in self.model.named_parameters():
            if name in self.trace_param_list:
                self.traces[name] = torch.zeros(
                    param.size(), dtype=torch.float32, device=device
                )

                self.grads[name] = torch.zeros(
                    param.size(), dtype=torch.float32, device=device
                )

            self.msgrads[name] = torch.zeros(
                param.size(), dtype=torch.float32, device=device
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

        # Compute targets
        trace_potential = V_curr + 0.5 * torch.log(pi[0, action] + self.min_denom)
        entropy = -torch.sum(torch.log(pi + self.min_denom) * pi)

        # Gradients to be combined with elig traces
        self.model.zero_grad()
        trace_potential.backward(retain_graph=True)
        self.store_current_trace_grads()

        # Update parameters except for on the first observation
        if last_state is not None:
            non_trace_loss = - self.entropy_beta * entropy
            self.model.zero_grad()
            non_trace_loss.backward()

            with torch.no_grad():
                # TD error
                V_last = self.model(last_state)[1]
                delta = (self.discount_gamma * (0 if is_terminal else V_curr)
                         + reward - V_last)

                # Update
                self.parameter_step(delta, time_step)

        # Accumulating trace (Always update trace)
        self.accumulate_eligibility_traces()

        # ==
        # Construct dict
        out_dict = None
        # TODO log the average trace magnitude?
        if last_state is not None:
            out_dict = {
                'value_loss': delta.item() ** 2,
            }

        return out_dict

    def parameter_step(self, trace_delta, time_step) -> None:
        """
        One step of parameter update with eligibility traces and RMSProp
        with initialization debiasing
        :param trace_delta: torch.tensor of the delta (error) to be
                            combined with the elig trace
        :param time_step: number of optim steps for initialization debiasing
        :return:
        """

        # Iterate over all model parameters
        for name, param in self.model.named_parameters():
            # Compute the trace for the current set of parameters
            trace = 0.0
            if name in self.traces:
                trace = self.traces[name]

            # Compute the parameter delta
            grad = (trace * trace_delta[0]) + (-param.grad)

            # Update the mean squared grad (for RMSProp)
            self.msgrads[name].data.copy_(
                self.grad_rms_gamma * self.msgrads[name]
                + ((1 - self.grad_rms_gamma) * grad * grad)
            )

            # Update RMSprop adaptive denominator
            delta_denom = torch.sqrt(
                self.msgrads[name] /
                (1 - self.grad_rms_gamma ** (time_step + 1))
                + self.grad_rms_eps
            )

            # Update model parameters
            self.model.state_dict()[name].copy_(
                param + self.lr_alpha * (grad / delta_denom)
            )

    def store_current_trace_grads(self) -> None:
        """
        Helper method, saves the current gradients on the parameters
        of self.model that require eligibility traces to self.grads
        :return: None
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.grads:
                    self.grads[name].data.copy_(param.grad)

    def accumulate_eligibility_traces(self) -> None:
        """
        Accumulate the gradients in self.grads in self.traces
        :return:
        """
        with torch.no_grad():
            for name in self.traces:
                self.traces[name].data.copy_(
                    self.trace_lambda * self.discount_gamma * self.traces[name]
                    + self.grads[name]
                )

    def clear_eligibility_traces(self) -> None:
        """
        Called to clear the elig traces at the end of an episode
        :return: None
        """
        for name in self.traces:
            self.traces[name].zero_()


if __name__ == '__main__':
    pass
