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
        # NOTE: i think this is basically like a sum of the losses used to compute the
        #       gradients
        trace_potential = (
                V_curr  # value grad
                + (0.5 * torch.log(pi[0, action] + self.min_denom))  # pol grad
        )

        # Save the value + policy loss + sf gradient (for traces)
        self.model.zero_grad()
        trace_potential.backward(retain_graph=True)
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), self.grads):
                # TODO check and confirm this is working
                grad.data.copy_(param.grad)

        #
        entropy = -torch.sum(torch.log(pi + self.min_denom) * pi)

        # Update parameters except for on the first observation
        if last_state is not None:

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

            # Inverse non-trace losses to optimize
            non_trace_inv_loss = - (
                    (0.5 * sf_loss) + (0.5 * rew_loss)
                    - (self.entropy_beta * entropy)
            )  # TODO add customizable coefficients
            self.model.zero_grad()
            non_trace_inv_loss.backward()

            with torch.no_grad():
                # ==
                # Errors

                # Value TD error
                V_last = self.model(last_state)[1]
                v_target = (self.discount_gamma *
                            (0 if is_terminal else lsf_V_curr) + reward)
                v_delta = v_target - V_last

                # ==
                # Update uses RMSProp with initialization debiasing
                for param, trace, ms_grad in zip(self.model.parameters(),
                                                 self.traces,
                                                 self.msgrads):
                    # elig trace deltas + other deltas
                    grad = trace * v_delta[0] + param.grad

                    # TODO NOTE: do a multiplication of trace * param.grad instead??

                    # RMSProp and update
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
        # TODO log the average trace magnitude
        if last_state is not None:
            out_dict = {
                'value_loss': v_delta.item() ** 2,
                'sf_loss': sf_loss.item(),
                'reward_loss': rew_loss.item(),
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
