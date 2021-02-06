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

        self.indiv_str_lr_dict = {}  # specified by user
        self.params_lr_dict = {}  # initialized

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
            # For traces
            if name in self.trace_param_list:
                self.traces[name] = torch.zeros(
                    param.size(), dtype=torch.float32, device=device
                )

                self.grads[name] = torch.zeros(
                    param.size(), dtype=torch.float32, device=device
                )
            # Mean squared grad for RMSProp
            self.msgrads[name] = torch.zeros(
                param.size(), dtype=torch.float32, device=device
            )

            # Populate the specific learning rate dictionary
            for parent_str in self.indiv_str_lr_dict:
                if name.startswith(parent_str):
                    self.params_lr_dict[name] = \
                        self.indiv_str_lr_dict[parent_str]

        print('self.params_lr_dict', self.params_lr_dict)
        print(self.model)  # TODO delete this and above?

    def get_action(self, state):
        return torch.multinomial(self.model(state)[0], 1)[0]

    def optimize_agent(self, sample, time_step):

        # ==
        # Unpack sample
        state = sample.state  # (batch_n=1, channel, height, width)
        next_state = sample.next_state  # (batch_n, c, h w)
        action = sample.action  # (batch_n=1, 1)
        reward = sample.reward  # (batch_n=1, 1)
        is_terminal = sample.is_terminal  # (batch_n=1, 1)

        pi, V_curr = self.model(state)

        # ==
        # Compute eligibility trace
        trace_potential = V_curr + 0.5 * torch.log(pi[0, action] + self.min_denom)
        # TODO: add sum for batch dim?

        # Save gradients for trace parameters
        self.model.zero_grad()
        trace_potential.backward(retain_graph=True)
        self.store_current_trace_grads()

        # Accumulating trace with stored gradients
        self.accumulate_eligibility_traces()

        # ==
        # Compute additional losses
        entropy = -torch.sum(torch.log(pi + self.min_denom) * pi)
        non_trace_loss = - self.entropy_beta * entropy
        # TODO: use a sum for batch dim?

        self.model.zero_grad()
        non_trace_loss.backward()

        # ==
        # Update parameters
        with torch.no_grad():
            # TD error
            V_next = self.model(next_state)[1]
            delta = (self.discount_gamma * (0 if is_terminal else V_next)
                     + reward - V_curr)

            # Update
            self.parameter_step(delta, time_step)

        # ==
        # Construct dict
        out_dict = None
        # TODO log the average trace magnitude?
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
            cur_param_lr = self.lr_alpha
            if name in self.params_lr_dict:
                cur_param_lr = self.params_lr_dict[name]

            self.model.state_dict()[name].copy_(
                param + cur_param_lr * (grad / delta_denom)
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


class ACLambdaOld(ACLambda):
    """
    Old reference code
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
        super().__init__(
            ModelCls, model_kwargs,
            discount_gamma=discount_gamma,
            lr_alpha=lr_alpha,
            trace_lambda=trace_lambda,
            entropy_beta=entropy_beta,
            grad_rms_gamma=grad_rms_gamma,
            grad_rms_eps=grad_rms_eps,
            min_denom=min_denom
        )

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


if __name__ == '__main__':
    pass
