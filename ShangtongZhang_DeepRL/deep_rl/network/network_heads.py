#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *


class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_head(phi)
        return dict(q=q)


class SF_Function(nn.Module):
    """
    A single (possibly multi-layer) SF function
    """

    def __init__(self, feature_dim, sf_hidden_sizes=None):
        super(SF_Function, self).__init__()
        self.feature_dim = feature_dim
        self.sf_hidden_sizes = sf_hidden_sizes

        if sf_hidden_sizes is None or len(sf_hidden_sizes) == 0:
            sf_fn = nn.Linear(self.feature_dim, self.feature_dim,
                              bias=False)
            sf_fn.weight.data.copy_(torch.eye(self.feature_dim))
        else:
            sf_fn_layers_list = [
                nn.Linear(self.feature_dim, sf_hidden_sizes[0]),
                nn.ReLU(),
            ]
            for i in range(1, len(sf_hidden_sizes)):
                sf_fn_layers_list.extend([
                    nn.Linear(sf_hidden_sizes[i - 1], sf_hidden_sizes[i]),
                    nn.ReLU(),
                ])
            sf_fn_layers_list.extend([
                nn.Linear(sf_hidden_sizes[-1], self.feature_dim),
            ])
            sf_fn = nn.Sequential(*sf_fn_layers_list)

        self.fn = sf_fn

    def forward(self, x):
        """
        :param x: torch tensor of feature, size (N, *, d)
        :return: tensor of SF, size (N, *, d)
        """
        return self.fn(x)


class LQVanillaNet_sharePsiR(nn.Module, BaseNet):
    def __init__(self, output_dim, body,
                 sf_grad_to_phi=False,
                 reward_grad_to_phi=True):
        super(LQVanillaNet_sharePsiR, self).__init__()
        self.body = body

        self.q_fn = layer_init(
            nn.Linear(body.feature_dim, output_dim, bias=False)
        )
        self.sf_fn = identity_init(
            nn.Linear(body.feature_dim, body.feature_dim, bias=False)
        )
        # self.sf_fn = SF_Function(body.feature_dim, [body.feature_dim])

        self.reward_fn = layer_init(
            nn.Linear(body.feature_dim, 1, bias=False)
        )

        self.sf_grad_to_phi = sf_grad_to_phi
        self.reward_grad_to_phi = reward_grad_to_phi

        self.to(Config.DEVICE)

    def forward(self, x, sf_lambda=0.0):
        phi = self.body(tensor(x))  # (N, d)
        q = self.q_fn(phi)  # (N, |A|)

        # Reward estimate
        if self.reward_grad_to_phi:
            r_s = self.reward_fn(phi) # (N, 1)
        else:
            r_s = self.reward_fn(phi.detach())

        # Lambda successor feature estimate
        if self.sf_grad_to_phi:
            psi = self.sf_fn(phi)  # (N, d)
        else:
            psi = self.sf_fn(phi.detach())

        # Lambda Value Function estimate
        sf_v = self.q_fn(psi)  # (N, |A|)
        sf_r = self.reward_fn(psi)  # (N, 1)

        lamb_q = (((1 - sf_lambda) * sf_v)
                  + (sf_lambda * sf_r))  # (N, |A|)

        return dict(phi=phi, psi=psi, q=q, rew=r_s, lamb_q=lamb_q)


class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return dict(q=q)


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return dict(prob=prob, log_prob=log_prob)


class RainbowNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body, noisy_linear):
        super(RainbowNet, self).__init__()
        if noisy_linear:
            self.fc_value = NoisyLinear(body.feature_dim, num_atoms)
            self.fc_advantage = NoisyLinear(body.feature_dim, action_dim * num_atoms)
        else:
            self.fc_value = layer_init(nn.Linear(body.feature_dim, num_atoms))
            self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))

        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.noisy_linear = noisy_linear
        self.to(Config.DEVICE)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc_value.reset_noise()
            self.fc_advantage.reset_noise()
            self.body.reset_noise()

    def forward(self, x):
        phi = self.body(tensor(x))
        value = self.fc_value(phi).view((-1, 1, self.num_atoms))
        advantage = self.fc_advantage(phi).view(-1, self.action_dim, self.num_atoms)
        q = value + (advantage - advantage.mean(1, keepdim=True))
        prob = F.softmax(q, dim=-1)
        log_prob = F.log_softmax(q, dim=-1)
        return dict(prob=prob, log_prob=log_prob)


class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return dict(quantile=quantiles)


class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}


class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(torch.cat([phi, a], dim=1)))


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.phi_params = list(self.phi_body.parameters())

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters()) + self.phi_params
        self.actor_params.append(self.std)
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters()) + self.phi_params

        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'mean': mean,
                'v': v}


class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'v': v}


class TD3Net(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 ):
        super(TD3Net, self).__init__()
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_action(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2
