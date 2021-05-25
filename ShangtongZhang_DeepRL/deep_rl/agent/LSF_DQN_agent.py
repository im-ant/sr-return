#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import time

from torch.optim import RMSprop, Adam

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *


class LSF_DQNAgent(DQNAgent):
    def __init__(self, config,
                 sf_lambda_kwargs=None,
                 sf_target_net=True,
                 sf_optim_kwargs=None,
                 reward_optim_kwargs=None):
        DQNAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.sf_lambda_kwargs = sf_lambda_kwargs
        self.lambda_scheduler = LinearSchedule(**sf_lambda_kwargs)
        self.sf_target_net = sf_target_net

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())

        # Construct optimizer param groups
        param_groups = [
            {'params': self.network.body.parameters()},
            {'params': self.network.q_fn.parameters()},
            {'params': self.network.sf_fn.parameters(),
             **sf_optim_kwargs},
            {'params': self.network.reward_fn.parameters(),
             **reward_optim_kwargs},
        ]
        self.optimizer = config.optimizer_fn(param_groups)

        #
        self.actor.set_network(self.network)
        self.total_steps = 0

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        # Unpack states
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        #
        if self.total_steps < config.exploration_steps:
            cur_sf_lambda = self.sf_lambda_kwargs.start
        else:
            cur_sf_lambda = self.lambda_scheduler()

        # Compute next step estimate
        with torch.no_grad():
            next_dict = self.target_network(
                next_states, sf_lambda=cur_sf_lambda)

            # Next step SF estimate
            if self.sf_target_net:
                psi_next = next_dict['psi'].detach()  # (N, d)
            else:
                psi_next = self.network(
                    next_states, sf_lambda=cur_sf_lambda)['psi'].detach()

            # Next step Q estimate
            q_next = next_dict['lamb_q'].detach()  # (N, |A|)
            if self.config.double_q:
                best_actions = torch.argmax(self.network(
                    next_states, sf_lambda=cur_sf_lambda)['lamb_q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]  # (N, 1)
        #
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)

        # Current estimates
        cur_dict = self.network(states, sf_lambda=cur_sf_lambda)
        phi_cur = cur_dict['phi']
        psi_cur = cur_dict['psi']
        reward_cur = cur_dict['rew']

        actions = tensor(transitions.action).long()
        q_cur = cur_dict['q']
        q_cur = q_cur.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Targets
        q_target = (rewards + (self.config.discount ** config.n_step
                               * q_next * masks))

        psi_target = (
            phi_cur.detach() + (
                ((self.config.discount * cur_sf_lambda) ** config.n_step)
                * (psi_next * masks.unsqueeze(1))
            ))

        # Errors
        psi_error = psi_target - psi_cur
        rew_error = rewards - reward_cur
        q_error = q_target - q_cur

        return dict(psi=psi_error, rew=rew_error, q=q_error)

    def step(self):
        config = self.config
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            if config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()

            error_dict = self.compute_loss(transitions)
            q_error = error_dict['q']

            if isinstance(transitions, PrioritizedTransition):
                priorities = q_error.abs().add(config.replay_eps).pow(config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                q_error = q_error.mul(weights)

            q_loss = self.reduce_loss(q_error)
            psi_loss = self.reduce_loss(error_dict['psi'])
            rew_loss = self.reduce_loss(error_dict['rew'])
            total_loss = q_loss + psi_loss + rew_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

            # Logging
            self.logger.add_scalar(
                'Loss/total_loss', total_loss,
                self.total_steps)
            self.logger.add_scalar(
                'Loss/q_loss', q_loss,
                self.total_steps)
            self.logger.add_scalar(
                'Loss/psi_loss', psi_loss,
                self.total_steps)
            self.logger.add_scalar(
                'Loss/reward_loss', rew_loss,
                self.total_steps)

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
