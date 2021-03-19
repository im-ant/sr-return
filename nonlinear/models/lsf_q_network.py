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


class LQNet_shareQR(nn.Module):
    """
    Lambda Q function network
    Separate: SF branch for each action
    Share between actions: Q function layer, Reward function layer
    """

    def __init__(self, in_channels, num_actions, sf_hidden_sizes):
        super(LQNet_shareQR, self).__init__()

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
                                  out_features=1, bias=False)  # TODO decide whether action out

        self.reward_fn = nn.Linear(in_features=self.feature_dim,
                                   out_features=1, bias=False)  # TOO decide whether action out

    def forward(self, x, sf_lambda):
        phi = self.encoder(x)  # (N, *, d)
        sf_phi = self.sf_fn(phi)  # (N, *, |A|, d)

        sf_v = self.value_fn(sf_phi)  # (N, *, |A|, 1)
        sf_r = self.reward_fn(sf_phi)  # (N, *, |A|, 1)

        lsf_q = ((1 - sf_lambda) * sf_v) + (sf_lambda * sf_r)
        lsf_q = lsf_q.squeeze(-1)  # (N, *, |A|)

        return lsf_q


class LQNet_sharePsiR(nn.Module):
    """
    Lambda Q function network
    Separate: Q function for each action
    Share between actions: SF layer, Reward layer
    """

    def __init__(self, in_channels, num_actions, sf_hidden_sizes):
        super(LQNet_sharePsiR, self).__init__()

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

        self.sf_fn = SF_Function(feature_dim=self.feature_dim,
                                 sf_hidden_sizes=self.sf_hidden_sizes)

        self.value_fn = nn.Linear(in_features=self.feature_dim,
                                  out_features=self.num_actions,
                                  bias=False)

        self.reward_fn = nn.Linear(in_features=self.feature_dim,
                                   out_features=1, bias=False)

    def forward(self, x, sf_lambda):
        phi = self.encoder(x)  # (N, *, d)
        sf_phi = self.sf_fn(phi)  # (N, *, d)

        sf_v = self.value_fn(sf_phi)  # (N, *, |A|)
        sf_r = self.reward_fn(sf_phi)  # (N, *, 1)

        lsf_q = (((1 - sf_lambda) * sf_v)
                 + (sf_lambda * sf_r))  # (N, *, |A|)
        return lsf_q


if __name__ == '__main__':
    model = SF_PerAction(num_actions=4, feature_dim=3, sf_hidden_sizes=[2, 2])
    print(model)
    pass
