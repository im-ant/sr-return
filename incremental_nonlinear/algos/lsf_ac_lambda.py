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
                 sf_lr=0.00048828125,
                 trace_lambda=0.0,
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

        # NOTE: somewhat hacky for now
        self.indiv_str_lr_dict = {
            'sf_fn': sf_lr,
        }

    def optimize_agent(self, sample, time_step):

        # ==
        # Unpack sample
        state = sample.state  # (batch_n=1, channel, height, width)
        next_state = sample.next_state  # (batch_n, c, h w)
        action = sample.action  # (batch_n=1, 1)
        reward = sample.reward  # (batch_n=1, 1)
        is_terminal = sample.is_terminal  # (batch_n=1, 1)

        model_out_tup = self.model.compute_pi_v_sf_r_lsfv(state,
                                                          self.sf_lambda)
        pi, V_curr, sf_curr, r_curr, lsf_V_curr, phi_curr = model_out_tup

        # ==
        # Compute eligibility trace
        trace_potential = V_curr + 0.5 * torch.log(pi[0, action] + self.min_denom)
        # TODO: add sum for batch dim?

        # Gradients to be combined with elig traces
        self.model.zero_grad()
        trace_potential.backward(retain_graph=True)
        self.store_current_trace_grads()

        # Accumulating trace with stored gradients
        self.accumulate_eligibility_traces()

        # ==
        # Compute additional losses
        # TODO for all below, sum across batch dimension?
        entropy = -torch.sum(torch.log(pi + self.min_denom) * pi)

        # SF loss
        with torch.no_grad():
            phi_next = self.model.compute_phi(next_state)
            sf_next = self.model.compute_sf_from_phi(phi_next)  # (1, d)
        sf_target = phi_curr.detach().clone() + (
                (self.sf_lambda * self.discount_gamma) * sf_next.detach()
        )  # TODO NOTE do I need to clone after detach?
        sf_loss = torch.mean(0.5 * (sf_target - sf_curr) ** 2)

        # Reward loss
        # rew_last = self.model.reward_layer(phi_last)[0]  # TODO delete
        rew_loss = 0.5 * (reward - r_curr) ** 2

        # Combine non-trace losses to optimize
        non_trace_loss = (
                (0.5 * sf_loss) + (0.5 * rew_loss)
                - (self.entropy_beta * entropy)
        )  # TODO add customizable coefficients
        self.model.zero_grad()
        non_trace_loss.backward()

        with torch.no_grad():
            # TD error
            _, _, _, _, lsf_V_next, _ = self.model.compute_pi_v_sf_r_lsfv(
                next_state, self.sf_lambda
            )

            lsf_V_next = self.model(next_state)[1]
            delta = (self.discount_gamma *
                     (0 if is_terminal else lsf_V_next)
                     + reward - V_curr)

            # Update
            self.parameter_step(delta, time_step)

            # For logging: compute difference in estimate
            lsf_v_theta_v_diff = torch.norm(
                (lsf_V_curr - V_curr)
            )

        # ==
        # Construct dict
        # TODO log the average trace magnitude?
        out_dict = {
            'value_loss': delta.item() ** 2,
            'sf_loss': sf_loss.item(),
            'reward_loss': rew_loss.item(),
            'lsf_v_v_diff': lsf_v_theta_v_diff.item(),
        }

        return out_dict


if __name__ == '__main__':
    pass
