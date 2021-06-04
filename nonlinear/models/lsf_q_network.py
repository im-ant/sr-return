# ============================================================================
# Modified from original MinAtar examples from authors:
# Kenny Young (kjyoung@ualberta.ca)
# Tian Tian (ttian@ualberta.ca)
#
# Anthony G. Chen
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as f


class ConvHead(nn.Module):
    """
    Convolution head for the image encoding
    By default: One hidden 2D conv with variable number of input channels.
    16 filters, a quarter of the original DQN paper of 64.  One hidden fully
    connected linear layer with a quarter of the original DQN paper of 512
    rectified units.
    """

    def __init__(self, in_channels, feature_dim=128):
        super(ConvHead, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=feature_dim)

    def forward(self, x):
        # Rectified output from the first conv layer
        x = f.relu(self.conv(x))

        # Rectified output from hidden layer
        return f.relu(self.fc_hidden(x.view(x.size(0), -1)))


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


class SF_PerAction(nn.Module):
    """
    Successor feature module
    """

    def __init__(self, feature_dim, num_actions, sf_hidden_sizes=None):
        super(SF_PerAction, self).__init__()

        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.sf_hidden_sizes = sf_hidden_sizes

        # Initialize layers, independent for each action
        self.sf_fn_list = nn.ModuleList([])
        for __ in range(self.num_actions):
            sf_fn = SF_Function(self.feature_dim, self.sf_hidden_sizes)
            self.sf_fn_list.append(sf_fn)

    def forward(self, x):
        """
        :param x: torch tensor of feature, size (N, *, d)
        :return: tensor of SF, size (N, *, |A|, d)
        """
        x_sf_list = []
        for sf_fn in self.sf_fn_list:
            x_sf = sf_fn(x)
            x_sf_list.append(x_sf.unsqueeze(-2))
        return torch.cat(x_sf_list, dim=-2)


def helper_sf_gather(sf_tensor, indeces):
    """
    Helper function, given a SF tensor and action indeces, return only the
    SFs associated with the actions
    :param sf_tensor: SF tensor with action specific dimension of -2, here
                      with size (batch_n, *, |A|, d)
    :param indeces: action indeces, size (batch_n, *, 1)
    :return: action-sf tensor of size (batch_n, *, d)
    """
    sizes = list(sf_tensor.size())  # list: [batch, *, |A|, d]
    sizes[-2] = 1  # list: [batch, *, 1, d]

    idxs = indeces.clone().unsqueeze(-1)  # (batch_n, 1, 1)
    while len(sizes) > len(idxs.size()):
        idxs = idxs.unsqueeze(-2)  # (batch_n, *(1), 1, 1)
    idxs = idxs.expand(sizes)  # (batch_n, *, 1, d)  # TODO check if this works with dim > 3

    # gathering. For dim == -2
    # out[i]...[j][k] = input[i]...[idxs[i]...[j][k]][k]
    sf_a_tensor = sf_tensor.gather(-2, idxs)  # (batch_n, *, 1, d)
    return sf_a_tensor.squeeze(-2)  # (batch_n, *, d)


class LQNet_sharePsiR(nn.Module):
    """
    Lambda Q function network
    Separate: Q function for each action
    Share between actions: SF layer, Reward layer
    """

    def __init__(self, in_channels, num_actions,
                 sf_hidden_sizes,
                 sf_grad_to_phi=False,
                 reward_grad_to_phi=True):
        super(LQNet_sharePsiR, self).__init__()

        # ==
        # Attributes
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.feature_dim = 128
        self.sf_hidden_sizes = sf_hidden_sizes

        self.sf_grad_to_phi = sf_grad_to_phi
        self.reward_grad_to_phi = reward_grad_to_phi

        # ==
        # Initialize modules
        self.encoder = ConvHead(in_channels=self.in_channels,
                                feature_dim=self.feature_dim)

        self.sf_fn = SF_Function(feature_dim=self.feature_dim,
                                 sf_hidden_sizes=self.sf_hidden_sizes)

        self.value_fn = nn.Linear(in_features=self.feature_dim,
                                  out_features=self.num_actions,
                                  bias=False)

        self.reward_fn = nn.Linear(in_features=self.feature_dim,
                                   out_features=1, bias=False)

    def compute_all_forward(self, x, sf_lambda):
        phi = self.encoder(x)  # (N, *, d)
        sf_phi = self.sf_fn(phi)  # (N, *, d)

        # Estimates based on SF
        sf_v = self.value_fn(sf_phi)  # (N, *, |A|)
        sf_r = self.reward_fn(sf_phi)  # (N, *, 1)

        lsf_qvec = (((1 - sf_lambda) * sf_v)
                    + (sf_lambda * sf_r))  # (N, *, |A|)

        # Estimate based on phi
        phi_qvec = self.value_fn(phi)  # (N, *, |A|)
        phi_r = self.reward_fn(phi)  # (N, *, 1)

        return phi, phi_qvec, phi_r, sf_phi, lsf_qvec

    def forward(self, x, sf_lambda):
        """Q vector used in the policy and in general when model is called"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        return out_tup[-1]  # lsf_qvec (N, *, |A|)

    def compute_estimates(self, x, actions, sf_lambda):
        """Quantites to estimate in forward pass"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        phi, phi_qvec, phi_r, sf_phi_hasGrad, lsf_qvec = out_tup

        # SF given current state, optional detach grad
        if self.sf_grad_to_phi:
            sf_phi = sf_phi_hasGrad
        else:
            sf_phi = self.sf_fn(phi.detach())  # (N, *, d)
        # Q given current state-action
        Q_s_a = phi_qvec.gather(-1, actions)  # (batch, *, 1)
        # Lambda Q function given current state-action
        lsfQ_s_a = lsf_qvec.gather(-1, actions)  # (batch, *, 1)
        # Reward given current state, optional detach grad
        if self.reward_grad_to_phi:
            R_s = phi_r  # (batch, *, 1)
        else:
            R_s = self.reward_fn(phi.detach())  # (batch, *, 1)

        return phi, sf_phi, Q_s_a, R_s, lsfQ_s_a

    def compute_targets(self, x, sf_lambda):
        """Quantities relevant for the bootstrap target"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        phi, phi_qvec, phi_r, sf_phi, lsf_qvec = out_tup

        # max Q values
        max_QAs = lsf_qvec.max(-1)  # values: (N,), indeces (N,)
        maxQ_val = max_QAs[0].unsqueeze(-1)  # max Q values, (N, 1)

        return phi, sf_phi, maxQ_val


class LQNet_sharePsiR_fwdQ(LQNet_sharePsiR):
    """
    Same as LQNet_sharePsiR but uses the Q function rather than the
    lambda Q function for the policy
    """

    def forward(self, x, sf_lambda):
        out_tup = self.compute_all_forward(x, sf_lambda)
        return out_tup[1]  # phi_qvec (N, *, |A|)


class LQNet_sharePsiR_gradQ_fwdQ(LQNet_sharePsiR):
    """
    Same as LQNet_sharePsiR, with changes:
        - Use the Q function (rather than the lambda Q) for policy
        - Reward gradient does not go into encoder layer (so only the
          Q gradient is propogated into the encoder)
    NOTE: deprecated class since adding the gradient passing arguments in
          class LQNet_sharePsiR
    """

    def compute_all_forward(self, x, sf_lambda):
        phi = self.encoder(x)  # (N, *, d)
        sf_phi = self.sf_fn(phi)  # (N, *, d)

        # Estimates based on SF
        sf_v = self.value_fn(sf_phi)  # (N, *, |A|)
        sf_r = self.reward_fn(sf_phi)  # (N, *, 1)

        lsf_qvec = (((1 - sf_lambda) * sf_v)
                    + (sf_lambda * sf_r))  # (N, *, |A|)

        # Estimate based on phi
        phi_qvec = self.value_fn(phi)  # (N, *, |A|)
        phi_r = self.reward_fn(phi.detach())  # (N, *, 1)

        return phi, phi_qvec, phi_r, sf_phi, lsf_qvec

    def forward(self, x, sf_lambda):
        """Q vector used in the policy and in general when model is called"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        return out_tup[-1]  # lsf_qvec (N, *, |A|)


class LQNet_shareQR(nn.Module):
    """
    Lambda Q function network
    Separate: SF branch for each action
    Share between actions: Q function layer, Reward function layer
    """

    def __init__(self, in_channels, num_actions,
                 sf_hidden_sizes,
                 sf_grad_to_phi=False,
                 value_grad_to_phi=True,
                 reward_grad_to_phi=True,
                 ):
        super(LQNet_shareQR, self).__init__()

        # ==
        # Attributes
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.feature_dim = 128
        self.sf_hidden_sizes = sf_hidden_sizes

        self.sf_grad_to_phi = sf_grad_to_phi
        self.value_grad_to_phi = value_grad_to_phi
        self.reward_grad_to_phi = reward_grad_to_phi

        # ==
        # Initialize modules
        self.encoder = ConvHead(in_channels=self.in_channels,
                                feature_dim=self.feature_dim)

        self.sf_fn = SF_PerAction(feature_dim=self.feature_dim,
                                  num_actions=self.num_actions,
                                  sf_hidden_sizes=self.sf_hidden_sizes)

        self.value_fn = nn.Linear(in_features=self.feature_dim,
                                  out_features=1, bias=False)

        self.reward_fn = nn.Linear(in_features=self.feature_dim,
                                   out_features=1, bias=False)

    def compute_all_forward(self, x, sf_lambda):
        phi = self.encoder(x)  # (N, *, d)
        sf_phi = self.sf_fn(phi)  # (N, *, |A|, d)

        # Estimates using SF
        sf_v = self.value_fn(sf_phi)  # (N, *, |A|, 1)
        sf_r = self.reward_fn(sf_phi)  # (N, *, |A|, 1)

        lsf_qvec = (((1 - sf_lambda) * sf_v)
                    + (sf_lambda * sf_r))  # (N, *, |A|, 1)
        lsf_qvec = lsf_qvec.squeeze(-1)  # (N, *, |A|)

        # Estimates using phi
        phi_qvec = self.value_fn(phi)  # (N, *, 1)
        phi_r = self.reward_fn(phi)  # (N, *, 1)

        return phi, phi_qvec, phi_r, sf_phi, lsf_qvec

    def forward(self, x, sf_lambda):
        """Q vector used in the policy and in general when model is called"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        return out_tup[-1]  # lsf_qvec (N, *, |A|)

    def compute_estimates(self, x, actions, sf_lambda):
        """Quantites to estimate in forward pass"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        phi, phi_qvec, phi_r, sf_phi_hasGrad, lsf_qvec = out_tup

        # SF givven current state-action
        if self.sf_grad_to_phi:
            sf_phi = sf_phi_hasGrad  # (N, *, |A|, d)
        else:
            sf_phi = self.sf_fn(phi.detach())  # (N, *, |A|, d)
        SF_s_a = helper_sf_gather(sf_phi, actions)  # (batch, *, d)

        # Q given current state
        if self.value_grad_to_phi:
            Q_s_a = phi_qvec  # (batch, *, 1)
        else:
            Q_s_a = self.value_fn(phi.detach())

        # Lambda Q function given current state-action
        lsfQ_s_a = lsf_qvec.gather(-1, actions)  # (batch, *, 1)

        # Reward given current state
        if self.reward_grad_to_phi:
            R_s = phi_r  # (batch, *, 1)
        else:
            R_s = self.reward_fn(phi.detach())

        return phi, SF_s_a, Q_s_a, R_s, lsfQ_s_a

    def compute_targets(self, x, sf_lambda):
        """Quantities relevant for the bootstrap target"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        phi, phi_qvec, phi_r, sf_phi, lsf_qvec = out_tup

        max_QAs = lsf_qvec.max(-1)  # values: (N,), indeces (N,)
        maxQ_acts = max_QAs[1].unsqueeze(-1)  # max Q action indeces, (N, 1)

        # SFs given max Q action
        maxQ_sf = helper_sf_gather(sf_phi, maxQ_acts)  # (N, d)
        # max Q values
        maxQ_val = max_QAs[0].unsqueeze(-1)  # max Q values, (N, 1)

        return phi, maxQ_sf, maxQ_val


class LQNet_shareR(nn.Module):
    """
    Lambda Q function network
    Separate: SF and Q functions for each action
    Share between actions: SF layer, Reward layer
    """

    def __init__(self, in_channels, num_actions, sf_hidden_sizes):
        super(LQNet_shareR, self).__init__()

        # ==
        # Attributes
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.feature_dim = 128
        self.sf_hidden_sizes = sf_hidden_sizes

        # ==
        # Initialize modules
        self.encoder = ConvHead(in_channels=self.in_channels,
                                feature_dim=self.feature_dim)

        self.sf_fn = SF_PerAction(feature_dim=self.feature_dim,
                                  num_actions=self.num_actions,
                                  sf_hidden_sizes=self.sf_hidden_sizes)

        self.value_fn = nn.Linear(in_features=self.feature_dim,
                                  out_features=self.num_actions,
                                  bias=False)

        self.reward_fn = nn.Linear(in_features=self.feature_dim,
                                   out_features=1, bias=False)

    def compute_all_forward(self, x, sf_lambda):
        phi = self.encoder(x)  # (N, *, d)
        sf_phi = self.sf_fn(phi)  # (N, *, |A|, d)

        all_a_sf_v = self.value_fn(sf_phi)  # (N, *, |A|, |A|)
        sf_v = torch.diagonal(
            all_a_sf_v, dim1=-2, dim2=-1
        )  # batch diagonal, (N, *, |A|)

        sf_r = self.reward_fn(sf_phi)  # (N, *, |A|, 1)
        sf_r = sf_r.squeeze(-1)  # (N, *, |A|)

        lsf_qvec = ((1 - sf_lambda) * sf_v) \
                   + (sf_lambda * sf_r)  # (N, *, |A|)

        # normal items
        phi_qvec = self.value_fn(phi)  # (N, *, |A|)
        phi_r = self.reward_fn(phi)  # (N, *, 1)

        return phi, phi_qvec, phi_r, sf_phi, lsf_qvec

    def forward(self, x, sf_lambda):
        """Q vector used in the policy and in general when model is called"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        return out_tup[-1]  # lsf_qvec (N, *, |A|)

    def compute_estimates(self, x, actions, sf_lambda):
        """Quantites to estimate in forward pass"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        phi, phi_qvec, phi_r, sf_phi, lsf_qvec = out_tup

        # SF given current state-action, detach gradient from features
        de_sf = self.sf_fn(phi.detach())  # (N, *, |A|, d)
        SF_s_a = helper_sf_gather(de_sf, actions)  # (batch, *, d)
        # Q given current state-action
        Q_s_a = phi_qvec.gather(-1, actions)  # (batch, *, 1)
        # Lambda Q function given current state-action
        lsfQ_s_a = lsf_qvec.gather(-1, actions)  # (batch, *, 1)
        # Reward given current state
        R_s = phi_r  # (batch, *, 1)

        return phi, SF_s_a, Q_s_a, R_s, lsfQ_s_a

    def compute_targets(self, x, sf_lambda):
        """Quantities relevant for the bootstrap target"""
        out_tup = self.compute_all_forward(x, sf_lambda)
        phi, phi_qvec, phi_r, sf_phi, lsf_qvec = out_tup

        max_QAs = lsf_qvec.max(-1)  # values: (N,), indeces (N,)
        maxQ_acts = max_QAs[1].unsqueeze(-1)  # max Q action indeces, (N, 1)

        # SFs given max Q action
        maxQ_sf = helper_sf_gather(sf_phi, maxQ_acts)  # (N, d)
        # max Q values
        maxQ_val = max_QAs[0].unsqueeze(-1)  # max Q values, (N, 1)

        return phi, maxQ_sf, maxQ_val


if __name__ == '__main__':
    model = SF_PerAction(num_actions=4, feature_dim=3, sf_hidden_sizes=[2, 2])
    print(model)
    pass
