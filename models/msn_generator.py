# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import cuda.MDS.MDS_module as MDS_module
import cuda.expansion_penalty.expansion_penalty_module as expansion
from models.atlasnet_generator import PointEncoder
from models.sparenet_generator import PointGenCon, PointNetRes


class MSN(nn.Module):
    def __init__(
        self,
        num_points=16382,
        bottleneck_size=1024,
        n_primitives=32,
    ):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.expansion = expansion.expansionPenaltyModule()

        self.encoder = PointEncoder(bottleneck_size=self.bottleneck_size)

        self.decoder = nn.ModuleList(
            [
                PointGenCon(
                    input_dim=2 + self.bottleneck_size,
                    bottleneck_size=2 + self.bottleneck_size,
                    use_SElayer=False,
                )
                for i in range(self.n_primitives)
            ]
        )

        self.res = PointNetRes(use_SElayer=False)

    def forward(self, data):
        x = data["partial_cloud"]
        x = x.transpose(1, 2).contiguous()
        partial = x
        style = self.encoder(x)

        outs = []
        for i in range(self.n_primitives):
            rand_grid = Variable(
                torch.cuda.FloatTensor(
                    x.size(0), 2, self.num_points // self.n_primitives
                )
            )
            rand_grid.data.uniform_(0, 1)
            y = (
                style.unsqueeze(2)
                .expand(style.size(0), style.size(1), rand_grid.size(2))
                .contiguous()
            )
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        outs = torch.cat(outs, 2).contiguous()
        coarse = outs.transpose(1, 2).contiguous()
        dist, _, mean_mst_dis = self.expansion(
            coarse, self.num_points // self.n_primitives, 1.5
        )
        loss_mst = torch.mean(dist)
        id0 = torch.zeros(outs.shape[0], 1, outs.shape[2]).cuda().contiguous()
        outs = torch.cat((outs, id0), 1)
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2]).cuda().contiguous()
        partial = torch.cat((partial, id1), 1)
        xx = torch.cat((outs, partial), 2)
        resampled_idx = MDS_module.minimum_density_sample(
            xx[:, 0:3, :].transpose(1, 2).contiguous(), coarse.shape[1], mean_mst_dis
        )
        xx = MDS_module.gather_operation(xx, resampled_idx)
        delta = self.res(xx)
        xx = xx[:, 0:3, :]
        refine = (xx + delta).transpose(2, 1).contiguous()
        return coarse, refine, loss_mst
