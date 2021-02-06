# ============================================================================
# Categorical PG agent.
#
# Inherits from the CategoricalPgAgent:
# https://github.com/astooke/rlpyt/blob/master/rlpyt/agents/pg/categorical.py
#
# with added method from the AtariMixin agent:
# https://github.com/astooke/rlpyt/blob/master/rlpyt/agents/pg/atari.py#L9
#
# Author: Anthony G. Chen
# ============================================================================

import torch


from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method


class CategoricalPgLsfAgent(CategoricalPgAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution
    and LSF.
    """

    def make_env_to_model_kwargs(self, env_spaces):
        """
        Copied from the AtariMixin class, from
        rlpyt/agents/pg/atari.py
        """
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

    def __call__(self, observation, prev_action, prev_reward):
        """
        Copied from CategoricalPgAgent agent class,
        from rlpyt/agents/pg/categorical.py
        """
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def lsf_train_call(self, observation, prev_action, prev_reward, sf_lambda):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)

        pi, value, sf, rew, bs_value, phi = self.model.compute_pi_v_sf_r_lsfv(*model_inputs, sf_lambda)
        return buffer_to((DistInfo(prob=pi), value, sf, rew, bs_value, phi), device="cpu")
