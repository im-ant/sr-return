# ============================================================================
# Categorical policy model.
#
# Adopted from the pg atari_ff_model.py:
# https://github.com/astooke/rlpyt/blob/f04f23db1eb7b5915d88401fca67869968a07a37/rlpyt/models/pg/atari_ff_model.py
#
# Author: Anthony G. Chen
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel


class CategoricalPgLsfFfModel(torch.nn.Module):
    """
    Feedforward model for categorical agents: a convolutional network feeding an
    MLP with outputs for action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
    ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [16],
            kernel_sizes=kernel_sizes or [2],
            strides=strides or [1],
            paddings=paddings or [0],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        self.pi = torch.nn.Linear(self.conv.output_size, output_size)
        self.value_layer = torch.nn.Linear(self.conv.output_size, 1)  # TODO note sure if bias should be here or not

        self.sf_fn = nn.Sequential(
            nn.Linear(self.conv.output_size, self.conv.output_size),
            nn.ReLU(),
            nn.Linear(self.conv.output_size, self.conv.output_size),
            nn.ReLU(),
            nn.Linear(self.conv.output_size, self.conv.output_size),
        )
        #self.sf_layer = torch.nn.Linear(self.conv.output_size, self.conv.output_size, bias=False)
        #self.sf_layer.weight.data.copy_(torch.eye(self.conv.output_size))  # identity init

        self.reward_layer = torch.nn.Linear(self.conv.output_size, 1)

    def process_image(self, image):
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        #img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        img = img.mul_(1. / 15.)  # assuem range?range
        return img

    def forward(self, image, prev_action, reward):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        img = self.process_image(image)

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value_layer(fc_out).squeeze(-1)  # TODO change this?

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v

    def compute_pi_v_sf_r_lsfv(self, image, prev_action, reward, sf_lambda):
        img = self.process_image(image)

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)  # policy dist
        v = self.value_layer(fc_out).squeeze(-1)  # value estimate
        sf = self.sf_fn(fc_out)  # sf NOTE no need to squeeze (I think)  TODO detach or no?
        r = self.reward_layer(fc_out).squeeze(-1)  # reward

        # ==
        # The LSV bootstrap target
        with torch.no_grad():
            cp_sf = sf.detach().clone()
            sf_v = self.value_layer(cp_sf).squeeze(-1)
            sf_r = self.reward_layer(cp_sf).squeeze(-1)

            lsf_v = ((1 - sf_lambda) * sf_v) + (sf_lambda * sf_r)
            lsf_v = lsf_v.detach().clone()

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v, sf, r, lsf_v, phi = restore_leading_dims(
            (pi, v, sf, r, lsf_v, fc_out),
            lead_dim, T, B)

        return pi, v, sf, r, lsf_v, phi
