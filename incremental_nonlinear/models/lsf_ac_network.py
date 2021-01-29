##############################################################################
# ACNetwork
#
# Original Authors:
# Kenny Young (kjyoung@ualberta.ca)
# Tian Tian (ttian@ualberta.ca)
#
# References used for this implementation:
#   https://pytorch.org/docs/stable/nn.html#
#   https://pytorch.org/docs/stable/torch.html
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#
# Anthony G. Chen
##############################################################################


import torch
import torch.nn as nn
import torch.nn.functional as f

from models.ac_network import ACNetwork

# ==
# activation functions
dSiLU = lambda x: torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))
SiLU = lambda x: x * torch.sigmoid(x)


class nn_SiLU(nn.Module):
    def __init__(self):
        super(nn_SiLU, self).__init__()
    def forward(self, x):
        return SiLU(x)


# ==
# Network class
class LSF_ACNetwork(ACNetwork):
    def __init__(
            self,
            in_channels, num_actions,  # TODO use image_shape, output_size??
            fc_sizes=128,
            sf_hidden_sizes=None,
            detach_sf_grad=False,
    ):
        super().__init__(in_channels, num_actions, fc_sizes)

        # ==
        # Add the sf layers

        # Value layer without bias
        self.value = nn.Linear(in_features=self.fc_sizes,
                               out_features=1, bias=False)

        # Initialize reward function layer
        self.reward_layer = torch.nn.Linear(self.fc_sizes, 1, bias=False)

        # Initialize the SF function layer(s)
        if sf_hidden_sizes is None or len(sf_hidden_sizes) == 0:
            self.sf_fn = nn.Linear(self.fc_sizes, self.fc_sizes, bias=False)
            self.sf_fn.weight.data.copy_(torch.eye(self.fc_sizes))
        else:
            sf_fn_layers_list = [
                nn.Linear(self.fc_sizes, sf_hidden_sizes[0]),
                nn_SiLU(),
            ]
            for i in range(1, len(sf_hidden_sizes)):
                sf_fn_layers_list.extend([
                    nn.Linear(sf_hidden_sizes[i - 1], sf_hidden_sizes[i]),
                    nn_SiLU(),
                ])
            sf_fn_layers_list.extend([
                nn.Linear(sf_hidden_sizes[-1], self.fc_sizes),
            ])
            self.sf_fn = nn.Sequential(*sf_fn_layers_list)

        # Other attributes
        self.detach_sf_grad = detach_sf_grad

    def compute_phi(self, x):
        """Feature extractor"""
        # Output from the first conv with sigmoid linear activation
        x = SiLU(self.conv(x))
        # Output from the final hidden layer with derivative of sigmoid linear activation
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))
        return x

    def compute_sf_from_phi(self, phi):
        """Compute SF from feature"""
        if self.detach_sf_grad:
            return self.sf_fn(phi.detach().clone())
        else:
            return self.sf_fn(phi)

    def forward(self, x):
        # Feature extract
        x = self.compute_phi(x)

        # Return policy and value outputs
        return f.softmax(self.policy(x), dim=1), self.value(x)

    def compute_pi_v_sf_r_lsfv(self, x, sf_lambda):
        # Feature extract
        x = self.compute_phi(x)

        # Policy distribution and value outputs
        pi = f.softmax(self.policy(x), dim=1)
        v = self.value(x)

        # Successor feature
        sf = self.compute_sf_from_phi(x)

        # Reward
        r = self.reward_layer(x)

        # ==
        # The LSV bootstrap target
        with torch.no_grad():
            cp_sf = sf.detach().clone()
            sf_v = self.value(cp_sf)
            sf_r = self.reward_layer(cp_sf)

            lsf_v = ((1 - sf_lambda) * sf_v) + (sf_lambda * sf_r)
            lsf_v = lsf_v.detach().clone()

        return pi, v, sf, r, lsf_v, x


if __name__ == '__main__':
    model = ACNetwork(3, 4)
    print(model)
    pass
