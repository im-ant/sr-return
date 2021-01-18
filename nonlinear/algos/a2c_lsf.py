# ============================================================================
# A2C algorithm adopted to optimize the lambda-SF losses.
#
# Adopted from:
# https://github.com/astooke/rlpyt/blob/f04f23db1eb7b5915d88401fca67869968a07a37/rlpyt/algos/pg/a2c.py
#
# Author: Anthony G. Chen
# ============================================================================
from collections import namedtuple
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_method
from rlpyt.algos.utils import (discount_return, generalized_advantage_estimation,
                               valid_from_done)  # TODO maybe delete?

# ==
# Overwrite parent logging
OptInfo = namedtuple("OptInfo", [
    "loss", "gradNorm", "entropy", "perplexity",
    "value_loss", "sf_loss", "reward_loss",
])


# ==
# Class
class A2C_LSF(PolicyGradientAlgo):
    """
    Advantage Actor Critic algorithm (synchronous).  Trains the agent by
    taking one gradient step on each iteration of samples, with advantages
    computed by generalized advantage estimation.
    """

    # Overriding parent log fields
    opt_info_fields = tuple(f for f in OptInfo._fields)

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=0.5,
            sf_loss_coeff=0.5,
            reward_loss_coeff=0.5,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            sf_lambda=0.0,
            normalize_advantage=False,
    ):
        """Saves the input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size  # For logging.

    def optimize_agent(self, itr, samples):
        """
        Train the agent on input samples, by one gradient step.
        """

        """ TODO DELETE TEST PRINTS
        for n, p in self.agent.model.named_parameters():  # TODO delete
            print(n, p.data.size())
        for g in self.optimizer.param_groups:
            for p in g['params']:
                print(print(p.data.size()))
        a = 1/0
        """

        if hasattr(self.agent, "update_obs_rms"):
            # NOTE: suboptimal--obs sent to device here and in agent(*inputs).
            self.agent.update_obs_rms(samples.env.observation)
        self.optimizer.zero_grad()
        loss, entropy, perplexity, log_dict = self.loss(samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # ==
        # Logging items
        opt_info = OptInfo(
            loss=loss.item(),
            gradNorm=torch.tensor(grad_norm).item(),  # backwards compatible,
            entropy=entropy.item(),
            perplexity=perplexity.item(),
            **log_dict,
        )
        self.update_counter += 1
        return opt_info

    def loss(self, samples):
        """
        Computes the training loss: policy_loss + value_loss + entropy_loss.
        Policy loss: log-likelihood of actions * advantages
        Value loss: 0.5 * (estimated_value - return) ^ 2
        Organizes agent inputs from training samples, calls the agent instance
        to run forward pass on training data, and uses the
        ``agent.distribution`` to compute likelihoods and entropies.

        NOTE: currently only valid for feedforward agents.
              May implement recurrence following rlpyt/algos/pg/a2c.py
        """
        if self.agent.recurrent:
            raise NotImplementedError

        # ==
        # Get agent estimates
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,  # dummy
        )
        dist_info, value, sf, rew, bs_value, phi = self.agent.lsf_train_call(
            *agent_inputs, self.sf_lambda)

        # ==
        # Compute target estimates
        # TODO: try to compute everyone on device.
        # return_, advantage, valid = self.process_returns(samples)

        v_return, sf_return, advantage, valid = self.process_lsf_returns(
            samples, bs_value,
            phi.detach().clone(), sf.detach().clone(),
        )

        reward_target = samples.env.reward

        # ==
        # Compute losses

        # Policy loss
        dist = self.agent.distribution
        logli = dist.log_likelihood(samples.agent.action, dist_info)
        pi_loss = - valid_mean(logli * advantage, valid)

        # Value error
        value_error = 0.5 * (value - v_return) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        # SF error
        sf_error = 0.5 * (sf - sf_return) ** 2
        # print(sf_error[0,0,0:10]) ## TODO delete
        sf_loss = self.sf_loss_coeff * valid_mean(sf_error, valid)

        # Reward error
        rew_error = 0.5 * (rew - reward_target) ** 2
        rew_loss = self.reward_loss_coeff * torch.mean(rew_error)  # NOTE: always valid?

        # Policy entropy
        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + sf_loss + rew_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)

        # ==
        # Construct logs
        log_dict = {
            'value_loss': value_loss.item(),
            'sf_loss': sf_loss.item(),
            'reward_loss': rew_loss.item(),
        }

        return loss, entropy, perplexity, log_dict

    def process_lsf_returns(self, samples, bvs, phi, sfs):
        """
        Assumes all inputs have no associated gradients
        NOTE: deleted mid_batch_reset, recurrent and advantage normalization.
              for these features see:
              https://github.com/astooke/
              rlpyt/blob/f04f23db1eb7b5915d88401fca67869968a07a37/rlpyt/algos/pg/base.py#L41

        :param samples:
        :param bvs:
        :param phi:
        :param sfs:
        :return:
        """
        # Unpack values
        reward, done, value = (samples.env.reward, samples.env.done,
                               samples.agent.agent_info.value)
        done = done.type(reward.dtype)

        # ==
        # Construct one-step returns
        def compute_one_step_return(cumulant, prediction, discount, nd):
            """
            Assumes first dimension is time with the same index
            :param cumulant:
            :param prediction:
            :param discount:
            :param nd: not done
            :return:
            """
            # Assume first 2 dims are matched and expand rest
            while len(prediction.size()) > len(nd.size()):
                nd = nd.unsqueeze(-1)
            # Compute return
            ret_ = prediction.clone()
            ret_[:-1] = (cumulant[:-1] +
                         (discount * (prediction[:-1] * nd[1:])))

            # NOTE TODO this is a hack, only works if the last item is
            # end-of-episode, if not it is not valid
            ret_[-1] = cumulant[-1]

            return ret_

        if self.gae_lambda > 0:
            raise NotImplementedError
        else:
            # Value return and advantage
            v_return = compute_one_step_return(
                reward, bvs, self.discount, (1 - done)
            )
            advantage = v_return - value  # one step advantage

            # SF return
            sf_return = compute_one_step_return(
                phi, sfs, (self.discount * self.sf_lambda), (1 - done)
            )

            valid = torch.ones(done.size())
            valid[-1] *= done[-1]

        return v_return, sf_return, advantage, valid
