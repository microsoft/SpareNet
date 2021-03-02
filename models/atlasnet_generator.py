# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.sparenet_generator import PointGenCon, PointNetfeat


class AtlasNet(nn.Module):
    def __init__(
        self,
        num_points: int = 16382,
        bottleneck_size: int = 1024,
        n_primitives: int = 32,
    ):
        super(AtlasNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives

        self.encoder = PointEncoder(bottleneck_size=self.bottleneck_size)
        self.decoder = nn.ModuleList([PointGenCon(
                    input_dim=2 + self.bottleneck_size,
                    bottleneck_size=2 + self.bottleneck_size,
                    use_SElayer=False,
                ) for i in range(self.n_primitives)])

    def forward(self, data):
        x = data["partial_cloud"]
        x = x.transpose(1, 2).contiguous()  # [bs, 3, in_points]

        style = self.encoder(x)  # [batch_size, 1024]

        outs = []
        for i in range(0, self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
            rand_grid.data.uniform_(0, 1)  # [batch_size, 2, 512]
            y = style.unsqueeze(2).expand(style.size(0), style.size(1), rand_grid.size(2)).contiguous()  # [batch_size, 1024, 512], repeat 512 times
            y = torch.cat((rand_grid, y), 1).contiguous()  # [batch_size, 1026, 512]
            outs.append(self.decoder[i](y))  # append each [batch_size, 3, 512]
        outs = torch.cat(outs, 2).contiguous()  # [batch_size, 3, out_points] coarse output
        out1 = outs.transpose(1, 2).contiguous()  # [batch_size, out_points, 3]
        return out1


class PointEncoder(nn.Module):
    def __init__(self, bottleneck_size=1024, hide_size=1024):
        super(PointEncoder, self).__init__()
        self.feat_extractor = PointNetfeat(global_feat=True, use_SElayer=False, hide_size=hide_size)
        self.linear = nn.Linear(hide_size, bottleneck_size)
        self.bn = nn.BatchNorm1d(bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    partial = torch.randn((8, 5000, 3), device="cuda")
    gt = torch.randn((8, 8192, 3), device="cuda")

    #
    network = AtlasNet()
    network.cuda()
    #
    output = network(partial.transpose(2, 1).contiguous())
    print(output.size())