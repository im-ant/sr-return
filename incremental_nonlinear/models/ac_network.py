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

# ==
# activation functions
dSiLU = lambda x: torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))
SiLU = lambda x: x * torch.sigmoid(x)


# ==
# Network class
class ACNetwork(nn.Module):
    """
    Setup the AC-network with one hidden 2D conv with variable number of input
    channels. We use 16 filters, a quarter of the original DQN paper of 64.
    One hidden fully connected linear layer with a quarter of the original
    DQN paper of 512 rectified units. Finally, we use one output layer which
    is a fully connected softmax layer with a single output for each valid
    action for the policy network, and another output which is a fully
    connected linear layer, with a single output for the state value.
    """
    def __init__(self,
                 in_channels, num_actions,
                 fc_sizes=128):
        # TODO use image_shape,
        #             output_size??
        super(ACNetwork, self).__init__()

        self.fc_sizes = fc_sizes

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        # TODO: change this to rlpyt's conv head?

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units,
                                   out_features=self.fc_sizes)

        # Output layer:
        self.policy = nn.Linear(in_features=self.fc_sizes,
                                out_features=num_actions)
        self.value = nn.Linear(in_features=self.fc_sizes,
                               out_features=1)

    # As per implementation instructions, the forward function should be overwritten by all subclasses
    def forward(self, x):
        # Output from the first conv with sigmoid linear activation
        x = SiLU(self.conv(x))

        # Output from the final hidden layer with derivative of sigmoid linear activation
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))

        # Return policy and value outputs
        return f.softmax(self.policy(x), dim=1), self.value(x)


if __name__ == '__main__':
    model = ACNetwork(3, 4)
    print(model)
    pass
