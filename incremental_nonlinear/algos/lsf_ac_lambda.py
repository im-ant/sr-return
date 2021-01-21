# ============================================================================
# Adopted from original Authors:
# Kenny Young (kjyoung@ualberta.ca)
# Tian Tian (ttian@ualberta.ca)
#
# Anthony G. Chen
# ============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from algos.ac_lambda import ACLambda


class LSF_ACLambda(ACLambda):
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
                 min_denom=0.0001,
                 sf_lambda=0.0,
                 ):
        # TODO: put more things as arguments
        super().__init__(
            ModelCls, model_kwargs, discount_gamma=discount_gamma,
            lr_alpha=lr_alpha, trace_lambda=trace_lambda,
            entropy_beta=entropy_beta, grad_rms_gamma=grad_rms_gamma,
            grad_rms_eps=grad_rms_eps, min_denom=min_denom,
        )
        self.sf_lambda = sf_lambda

    def optimize_agent(self, sample, time_step):

        # states, next_states: (1, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal: (1, 1)
        last_state = sample.last_state
        state = sample.state
        action = sample.action
        reward = sample.reward
        is_terminal = sample.is_terminal

        model_out_tup = self.model.compute_pi_v_sf_r_lsfv(state,
                                                          self.sf_lambda)
        pi, V_curr, sf_curr, r_curr, lsf_V_curr, phi_curr = model_out_tup

        # Compute the targets
        trace_potential = V_curr + 0.5 * torch.log(pi[0, action] + self.min_denom)
        entropy = -torch.sum(torch.log(pi + self.min_denom) * pi)

        # Gradients to be combined with elig traces
        self.model.zero_grad()
        trace_potential.backward(retain_graph=True)
        self.store_current_trace_grads()

        # Update parameters except for on the first observation
        if last_state is not None:
            # ==
            # More losses

            # SF loss
            phi_last = self.model.compute_phi(last_state)
            sf_last = self.model.sf_fn(phi_last)  # (1, d)
            sf_target = phi_last.detach().clone() + (
                    (self.sf_lambda * self.discount_gamma) * sf_curr.detach().clone()
            )  # TODO NOTE do I need to clone after detach?
            sf_loss = torch.mean(0.5 * (sf_target - sf_last) ** 2)

            # Reward loss  # TODO check if dimension is correct
            rew_last = self.model.reward_layer(phi_last)[0]
            rew_loss = 0.5 * (reward - rew_last) ** 2

            # Combine non-trace losses to optimize
            non_trace_loss = (
                    (0.5 * sf_loss) + (0.5 * rew_loss)
                    - (self.entropy_beta * entropy)
            )  # TODO add customizable coefficients
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
        # TODO log the average trace magnitude
        if last_state is not None:
            out_dict = {
                'value_loss': delta.item() ** 2,
                'sf_loss': sf_loss.item(),
                'reward_loss': rew_loss.item(),
            }

        return out_dict


if __name__ == '__main__':
    pass
