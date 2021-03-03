# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils
from torch.nn import Parameter


class PatchDiscriminator(nn.Module):
    """
    inputs:
    - img: b x (2*views) x [img_size, img_size]
    - feat: bool
    - y(label): b

    outputs:
    - validity: b
    - feat:
        - feat_1: b x num_dims(16) x [img_size, img_size](128)
        - feat_2: b x num_dims(32) x [img_size, img_size](64)
        - feat_3: b x num_dims(64) x [img_size, img_size](32)
        - feat_4: b x num_dims(128) x [img_size, img_size](16)
    """

    def __init__(self, img_shape: tuple = (2, 256, 256)):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(
            in_filters: int, out_filters: int, normalization: bool = True
        ):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                SpectralNorm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv1 = nn.Sequential(
            *discriminator_block(img_shape[0], 16, normalization=False),
        )
        self.conv2 = nn.Sequential(
            *discriminator_block(16, 32),
        )
        self.conv3 = nn.Sequential(
            *discriminator_block(32, 64),
        )
        self.conv4 = nn.Sequential(
            *discriminator_block(64, 128),
        )
        self.conv5 = nn.Sequential(
            *discriminator_block(128, 256),
        )
        self.conv6 = nn.Sequential(
            *discriminator_block(256, 512),
        )
        self.adv_layer = SpectralNorm(nn.Conv2d(512, 1, 3, padding=1, bias=False))

    def forward(self, img, feat=False, y=None):

        feat_1 = self.conv1(img)  # [bs, 16, 128, 128]
        feat_2 = self.conv2(feat_1)  # [bs, 32, 64, 64]
        feat_3 = self.conv3(feat_2)  # [bs, 64, 32, 32]
        feat_4 = self.conv4(feat_3)  # [bs, 128, 16, 16]
        feat_5 = self.conv5(feat_4)  # [bs, 256, 8, 8]
        feat_6 = self.conv6(feat_5)  # [bs, 512, 4, 4]
        validity = self.adv_layer(feat_6)  # [bs, 1, 4, 4]

        validity = F.avg_pool2d(validity, validity.size()[2:]).view(
            validity.size()[0], -1
        )

        if feat:
            return validity, [feat_1, feat_2, feat_3, feat_4]
        else:
            return validity


class ProjectionD(nn.Module):
    """
    inputs:
    - img: b x (2*views) x [img_size, img_size]
    - feat: bool
    - y(label): b

    outputs:
    - validity: b
    - feat:
        - feat_1: b x num_dims(16) x [img_size, img_size](128)
        - feat_2: b x num_dims(32) x [img_size, img_size](64)
        - feat_3: b x num_dims(64) x [img_size, img_size](32)
        - feat_4: b x num_dims(128) x [img_size, img_size](16)
    """

    def __init__(self, num_classes: int = 0, img_shape: tuple = (2, 256, 256)):
        super(ProjectionD, self).__init__()

        def discriminator_block(in_filters: int, out_filters: int, bn: bool = True):
            block = [
                SpectralNorm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv1 = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
        )
        self.conv2 = nn.Sequential(
            *discriminator_block(16, 32),
        )
        self.conv3 = nn.Sequential(
            *discriminator_block(32, 64),
        )
        feat_num = 128
        self.conv4 = nn.Sequential(
            *discriminator_block(64, feat_num),
        )

        ds_size = img_shape[1] // (2 ** 4) 
        self.adv_layer = utils.spectral_norm(nn.Linear(feat_num * ds_size ** 2, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, feat_num * ds_size ** 2)
            )
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.adv_layer.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, img, feat=False, y=None):
        feat_1 = self.conv1(img)  # [bs, 16, 128, 128]
        feat_2 = self.conv2(feat_1)  # [bs, 32, 64, 64]
        feat_3 = self.conv3(feat_2)  # [bs, 64, 32, 32]
        feat_4 = self.conv4(feat_3)  # [bs, 128, 16, 16]
        out = feat_4.view(feat_4.shape[0], -1)
        validity = self.adv_layer(out)
        if y is not None:
            validity += torch.sum(self.l_y(y) * out, dim=1, keepdim=True)
        if feat:
            return validity, [feat_1, feat_2, feat_3, feat_4]
        else:
            return validity


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
