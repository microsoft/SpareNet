# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cuda.MDS.MDS_module as MDS_module
import cuda.expansion_penalty.expansion_penalty_module as expansion


class SpareNetGenerator(nn.Module):
    """
    inputs:
    - data:
        -partical_cloud: b x npoints1 x num_dims
        -gtcloud: b x npoints2 x num_dims

    outputs:
    - coarse pcd: b x npoints2 x num_dims
    - middle pcd: b x npoints2 x num_dims
    - refine pcd: b x npoints2 x num_dims
    - loss_mst:
    """

    def __init__(
        self,
        n_primitives: int = 32,
        hide_size: int = 4096,
        bottleneck_size: int = 4096,
        num_points: int = 16382,
        use_SElayer: bool = False,
        use_AdaIn: str = "no_use",
        encode: str = "Pointfeat",
    ):
        super(SpareNetGenerator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.use_AdaIn = use_AdaIn
        self.hide_size = hide_size

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.encoder = SpareNetEncode(
            hide_size=self.hide_size,
            bottleneck_size=self.bottleneck_size,
            use_SElayer=use_SElayer,
            encode=encode,
        )
        self.decoder = SpareNetDecode(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            bottleneck_size=self.bottleneck_size,
            use_AdaIn=self.use_AdaIn,
            use_SElayer=use_SElayer,
        )
        self.refine = SpareNetRefine(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            use_SElayer=use_SElayer,
        )

    def forward(self, data):
        partial_x = data["partial_cloud"]
        partial_x = partial_x.transpose(1, 2).contiguous()  # [bs, 3, in_points]
        partial = partial_x  # [batch_size, 3, in_points]

        # encode
        style = self.encoder(partial_x)  # [batch_size, 1024]

        # decode
        outs = self.decoder(style, partial_x)
        coarse = outs.transpose(1, 2).contiguous()  # [batch_size, out_points, 3]

        # refine first time
        middle, loss_mst = self.refine(outs, partial, coarse)

        # refine second time
        outs_2 = middle.transpose(1, 2).contiguous()
        refine, _ = self.refine(outs_2, partial, middle)

        return coarse, middle, refine, loss_mst


class SpareNetEncode(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1

    output
    - feture:  one x feature_size
    """

    def __init__(
        self,
        bottleneck_size=4096,
        use_SElayer=False,
        encode="Pointfeat",
        hide_size=4096,
    ):
        super(SpareNetEncode, self).__init__()
        print(encode)
        if encode == "Residualnet":
            self.feat_extractor = EdgeConvResFeat(
                use_SElayer=use_SElayer, k=8, output_size=hide_size, hide_size=4096
            )
        else:
            self.feat_extractor = PointNetfeat(
                global_feat=True, use_SElayer=use_SElayer, hide_size=hide_size
            )
        self.linear = nn.Linear(hide_size, bottleneck_size)
        self.bn = nn.BatchNorm1d(bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EdgeConvResFeat(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1

    output
    - feture:  b x feature_size
    """

    def __init__(
        self,
        num_point: int = 16382,
        use_SElayer: bool = False,
        k: int = 8,
        hide_size: int = 2048,
        output_size: int = 4096,
    ):
        super(EdgeConvResFeat, self).__init__()
        self.use_SElayer = use_SElayer
        self.k = k
        self.hide_size = hide_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(6, self.hide_size // 16, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 8, kernel_size=1, bias=False
        )
        self.conv4 = nn.Conv2d(
            self.hide_size // 4, self.hide_size // 4, kernel_size=1, bias=False
        )
        self.conv5 = nn.Conv1d(
            self.hide_size // 2, self.output_size // 2, kernel_size=1, bias=False
        )

        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)

        if use_SElayer:
            self.se1 = SELayer(channel=self.hide_size // 16)
            self.se2 = SELayer(channel=self.hide_size // 16)
            self.se3 = SELayer(channel=self.hide_size // 8)
            self.se4 = SELayer(channel=self.hide_size // 4)

        self.bn1 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn2 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn3 = nn.BatchNorm2d(self.hide_size // 8)
        self.bn4 = nn.BatchNorm2d(self.hide_size // 4)
        self.bn5 = nn.BatchNorm1d(self.output_size // 2)

        self.resconv1 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.resconv2 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 8, kernel_size=1, bias=False
        )
        self.resconv3 = nn.Conv1d(
            self.hide_size // 8, self.hide_size // 4, kernel_size=1, bias=False
        )

    def forward(self, x):
        # x : [bs, 3, num_points]
        batch_size = x.size(0)
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)
            x = self.relu1(self.se1(self.bn1(self.conv1(x))))
            x1 = x.max(dim=-1, keepdim=False)[0]

            x2_res = self.resconv1(x1)
            x = get_graph_feature(x1, k=self.k)
            x = self.relu2(self.se2(self.bn2(self.conv2(x))))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.se3(self.bn3(self.conv3(x))))
            x3 = x.max(dim=-1, keepdim=False)[0]
            x3 = x3 + x3_res

            x4_res = self.resconv3(x3)
            x = get_graph_feature(x3, k=self.k)
            x = self.relu4(self.se4(self.bn4(self.conv4(x))))
        else:
            x = get_graph_feature(x, k=self.k)
            x = self.relu1(self.bn1(self.conv1(x)))
            x1 = x.max(dim=-1, keepdim=False)[0]

            x2_res = self.resconv1(x1)
            x = get_graph_feature(x1, k=self.k)
            x = self.relu2(self.bn2(self.conv2(x)))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.bn3(self.conv3(x)))
            x3 = x.max(dim=-1, keepdim=False)[0]
            x3 = x3 + x3_res

            x4_res = self.resconv3(x3)
            x = get_graph_feature(x3, k=self.k)
            x = self.relu4(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x4 = x4 + x4_res

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.relu5(self.bn5(self.conv5(x)))

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]
        x = torch.cat((x1, x2), 1)  # [bs, 4096]

        x = x.view(-1, self.output_size)
        return x


class PointNetfeat(nn.Module):
    """
    input
    - point_cloud： b x num_dims x npoints_1

    output
    - feture:  b x feature_size
    """

    def __init__(
        self, num_points=16382, global_feat=True, use_SElayer=False, hide_size=4096
    ):
        super(PointNetfeat, self).__init__()
        self.use_SElayer = use_SElayer
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, hide_size, 1)
        self.hide_size = hide_size
        if use_SElayer:
            self.se1 = SELayer1D(channel=64)
            self.se2 = SELayer1D(channel=128)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(hide_size)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]  # x: [batch_size, 3, num_points]
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.bn1(self.conv1(x)))  # x: [batch_size, 64, num_points]
            x = F.relu(self.bn2(self.conv2(x)))  # x: [batch_size, 128, num_points]
            x = self.bn3(self.conv3(x))  # x: [batch_size, 1024, num_points]
        x, _ = torch.max(x, 2)  # x: [batch_size, num_points]
        x = x.view(-1, self.hide_size)
        return x


class SpareNetDecode(nn.Module):
    """
    inputs:
    - style(feature): b x feature_size

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(
        self,
        num_points: int = 16382,
        n_primitives: int = 32,
        bottleneck_size: int = 4096,
        use_AdaIn: str = "no_use",
        use_SElayer: bool = False,
    ):
        super(SpareNetDecode, self).__init__()
        self.use_AdaIn = use_AdaIn
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.bottleneck_size = bottleneck_size
        self.grid = grid_generation(self.num_points, self.n_primitives)
        if use_AdaIn == "share":
            self.decoder = nn.ModuleList(
                [
                    StyleBasedAdaIn(
                        input_dim=2,
                        style_dim=self.bottleneck_size,
                        use_SElayer=use_SElayer,
                    )
                    for i in range(self.n_primitives)
                ]
            )

            # MLP to generate AdaIN parameters
            self.mlp = nn.Sequential(
                nn.Linear(self.bottleneck_size, self.bottleneck_size),
                nn.ReLU(),
                nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
            )
        elif use_AdaIn == "no_share":
            self.decoder = nn.ModuleList(
                [
                    AdaInPointGenCon(
                        input_dim=2,
                        style_dim=self.bottleneck_size,
                        use_SElayer=use_SElayer,
                    )
                    for i in range(self.n_primitives)
                ]
            )

        elif use_AdaIn == "no_use":
            self.decoder = nn.ModuleList(
                [
                    PointGenCon(
                        input_dim=2 + self.bottleneck_size, use_SElayer=use_SElayer
                    )
                    for i in range(self.n_primitives)
                ]
            )

    def forward(self, style, partial_x):
        outs = []
        if self.use_AdaIn == "share":
            adain_params = self.mlp(style)
            for i in range(self.n_primitives):
                regular_grid = torch.cuda.FloatTensor(self.grid[i])
                regular_grid = regular_grid.transpose(0, 1).contiguous().unsqueeze(0)
                regular_grid = regular_grid.expand(
                    partial_x.size(0), regular_grid.size(1), regular_grid.size(2)
                ).contiguous()
                regular_grid = ((regular_grid - 0.5) * 2).contiguous()

                outs.append(self.decoder[i](regular_grid, style, adain_params))
        elif self.use_AdaIn == "no_share":
            for i in range(self.n_primitives):
                regular_grid = torch.cuda.FloatTensor(self.grid[i])
                regular_grid = regular_grid.transpose(0, 1).contiguous().unsqueeze(0)
                regular_grid = regular_grid.expand(
                    partial_x.size(0), regular_grid.size(1), regular_grid.size(2)
                ).contiguous()
                regular_grid = ((regular_grid - 0.5) * 2).contiguous()

                outs.append(self.decoder[i](regular_grid, style))
        elif self.use_AdaIn == "no_use":
            for i in range(self.n_primitives):
                regular_grid = torch.cuda.FloatTensor(self.grid[i])
                regular_grid = regular_grid.transpose(0, 1).contiguous().unsqueeze(0)
                regular_grid = regular_grid.expand(
                    partial_x.size(0), regular_grid.size(1), regular_grid.size(2)
                ).contiguous()
                regular_grid = ((regular_grid - 0.5) * 2).contiguous()

                y = (
                    style.unsqueeze(2)
                    .expand(style.size(0), style.size(1), regular_grid.size(2))
                    .contiguous()
                )
                y = torch.cat((regular_grid, y), 1).contiguous()
                outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()


class StyleBasedAdaIn(nn.Module):
    """
    inputs:
    - content: b x (x,y) x (num_points / nb_primitives)
    - style(feature): b x feature_size
    - adain_params: b x parameter_size

    outputs:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 1026,
        style_dim: int = 1024,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
    ):
        super(StyleBasedAdaIn, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.dec = GridDecoder(
            self.input_dim, self.bottleneck_size, use_SElayer=use_SElayer
        )

    def forward(self, content, style, adain_params):
        assign_adain_params(adain_params, self.dec)
        return self.dec(content)


class AdaInPointGenCon(nn.Module):
    """
    inputs:
    - content: b x (x,y) x (num_points / nb_primitives)
    - style(feature): b x feature_size

    outputs:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 1026,
        style_dim: int = 1024,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
    ):
        super(AdaInPointGenCon, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.dec = GridDecoder(
            self.input_dim, self.bottleneck_size, use_SElayer=use_SElayer
        )

        # MLP to generate AdaIN parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim),
            nn.ReLU(),
            nn.Linear(self.style_dim, get_num_adain_params(self.dec)),
        )

    def forward(self, content, style):
        adain_params = self.mlp(style)
        assign_adain_params(adain_params, self.dec)
        return self.dec(content)


class PointGenCon(nn.Module):
    """
    inputs:
    - content: b x (x,y) x (num_points / nb_primitives)

    outputs:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 4098,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
        dropout: bool = False,
    ):
        self.input_dim = input_dim
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.input_dim, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(
            self.bottleneck_size // 2, self.bottleneck_size // 4, 1
        )
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.use_SElayer = use_SElayer
        if self.use_SElayer:
            self.se1 = SELayer1D(channel=self.bottleneck_size)
            self.se2 = SELayer1D(channel=self.bottleneck_size // 2)
            self.se3 = SELayer1D(channel=self.bottleneck_size // 4)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)
        self.dropout = dropout
        if self.dropout:
            self.drop1 = nn.Dropout(0.4)
            self.drop2 = nn.Dropout(0.4)
            self.drop3 = nn.Dropout(0.4)

    def forward(self, x):
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.drop1(x)

        if self.use_SElayer:
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
        else:
            x = F.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.drop2(x)

        if self.use_SElayer:
            x = F.relu(self.se3(self.bn3(self.conv3(x))))
        else:
            x = F.relu(self.bn3(self.conv3(x)))
        if self.dropout:
            x = self.drop3(x)
        x = self.conv4(x)  # [batch_size, 3, 512] 3 features(position) for 512 points
        return x


class SpareNetRefine(nn.Module):
    """
    inputs:
    - inps: b x npoints2 x num_dims
    - partial: b x npoints1 x num_dims
    - coarse: b x num_dims x npoints2

    outputs:
    - refine_result: b x num_dims x npoints2
    - loss_mst: float32
    """

    def __init__(
        self,
        n_primitives: int = 32,
        num_points: int = 16382,
        use_SElayer: bool = False,
    ):
        super(SpareNetRefine, self).__init__()
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.expansion = expansion.expansionPenaltyModule()
        self.edgeres = False
        if self.edgeres:
            self.residual = EdgeRes(use_SElayer=use_SElayer)
        else:
            self.residual = PointNetRes(use_SElayer=use_SElayer)

    def forward(self, inps, partial, coarse):
        dist, _, mean_mst_dis = self.expansion(
            coarse, self.num_points // self.n_primitives, 1.5
        )
        loss_mst = torch.mean(dist)
        id0 = torch.zeros(inps.shape[0], 1, inps.shape[2]).cuda().contiguous()

        inps = torch.cat((inps, id0), 1)  # [batch_size, 4, out_points]
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2]).cuda().contiguous()
        partial = torch.cat((partial, id1), 1)  # [batch_size, 4, in_points]
        base = torch.cat((inps, partial), 2)  # [batch_size, 4, out_points+ in_points]

        resampled_idx = MDS_module.minimum_density_sample(
            base[:, 0:3, :].transpose(1, 2).contiguous(), coarse.shape[1], mean_mst_dis
        )
        base = MDS_module.gather_operation(base, resampled_idx)

        delta = self.residual(base)  # [batch_size, 3, out_points]
        base = base[:, 0:3, :]  # [batch_size, 3, out_points]
        outs = base + delta
        refine_result = outs.transpose(2, 1).contiguous()  # [batch_size, out_points, 3]
        return refine_result, loss_mst


class PointNetRes(nn.Module):
    """
    input:
    - inp: b x (num_dims+id) x num_points

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(self, use_SElayer: bool = False):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.use_SElayer = use_SElayer
        if use_SElayer:
            self.se1 = SELayer1D(channel=64)
            self.se2 = SELayer1D(channel=128)
            self.se4 = SELayer1D(channel=512)
            self.se5 = SELayer1D(channel=256)
            self.se6 = SELayer1D(channel=128)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        npoints = x.size()[2]
        # x: [batch_size, 4, num_points]
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x  # [batch_size, 64, num_points]

        if self.use_SElayer:
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
        else:
            x = F.relu(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))  # [batch_size, 1024, num_points]
        x, _ = torch.max(x, 2)  # [batch_size, 1024]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 1088, num_points]
        if self.use_SElayer:
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
        else:
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))  # [batch_size, 3, num_points]
        return x


class EdgeRes(nn.Module):
    """
    input:
    - inp: b x (num_dims+id) x num_points

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(self, use_SElayer: bool = False):
        super(EdgeRes, self).__init__()
        self.k = 8
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.conv4 = torch.nn.Conv2d(2176, 512, kernel_size=1, bias=False)
        self.conv5 = torch.nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv6 = torch.nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv7 = torch.nn.Conv2d(256, 3, kernel_size=1, bias=False)

        self.use_SElayer = use_SElayer
        if use_SElayer:
            self.se1 = SELayer(channel=64)
            self.se2 = SELayer(channel=128)
            self.se4 = SELayer(channel=512)
            self.se5 = SELayer(channel=256)
            self.se6 = SELayer(channel=128)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(1024)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        npoints = x.size()[2]
        # x: [batch_size, 4, num_points]
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.se1(self.bn1(self.conv1(x))))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.bn1(self.conv1(x)))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.bn2(self.conv2(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]

        x = get_graph_feature(x, k=self.k)  # [bs, 256, num_points, k]
        x = self.bn3(self.conv3(x))  # [batch_size, 1024, num_points, k]
        x = x.max(dim=-1, keepdim=False)[0]  # [bs, 1024, num_points]

        x, _ = torch.max(x, 2)  # [batch_size, 1024]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 1088, num_points]

        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 1024, num_points, k]
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 1024, num_points, k]
            x = F.relu(self.bn5(self.conv5(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.bn6(self.conv6(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        x = get_graph_feature(x, k=self.k)  # [bs, 256, num_points, k]
        x = self.th(self.conv7(x))
        x = x.max(dim=-1, keepdim=False)[0]  # [bs, 3, num_points]
        return x


class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)

    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer1D(nn.Module):
    """
    input:
        x:(b, c, m)

    output:
        out:(b, c, m')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def grid_generation(num_points, nb_primitives):
    """
    inputs:
    - num_points: int
    - nb_primitives: int

    outputs:
    - 2D grid: nb_primitives * (num_points / nb_primitives) * 2
    """
    num_points = num_points / nb_primitives
    grain_x = 2 ** np.floor(np.log2(num_points) / 2) - 1
    grain_y = 2 ** np.ceil(np.log2(num_points) / 2) - 1

    vertices = []
    for i in range(int(grain_x + 1)):
        for j in range(int(grain_y + 1)):
            vertices.append([i / grain_x, j / grain_y])

    print("generating 2D grid")
    return [vertices for i in range(nb_primitives)]


def get_num_adain_params(model):
    """
    input:
    - model: nn.module

    output:
    - num_adain_params: int
    """
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


def assign_adain_params(adain_params, model):

    """
    inputs:
    - adain_params: b x parameter_size
    - model: nn.module

    function:
    assign_adain_params
    """
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


def knn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)

    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices

    if torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k: int = 20, idx=None):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    - idx: neighbor_idx

    outputs:
    - feature: b x npoints1 x (num_dims*2)
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class AdaptiveInstanceNorm1d(nn.Module):
    """
    input:
    - inp: (b, c, m)

    output:
    - out: (b, c, m')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class GridDecoder(nn.Module):
    """
    input:
    - x: b x (x,y) x (num_points / nb_primitives)

    output:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 2,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
        use_sine: bool = False,
    ):
        super(GridDecoder, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim

        self.use_sine = use_sine
        if not self.use_sine:
            self.conv1 = torch.nn.Conv1d(self.input_dim, self.bottleneck_size, 1)
            self.conv2 = torch.nn.Conv1d(
                self.bottleneck_size, self.bottleneck_size // 2, 1
            )
            self.conv3 = torch.nn.Conv1d(
                self.bottleneck_size // 2, self.bottleneck_size // 4, 1
            )
            self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
            self.th = nn.Tanh()
        else:
            first_omega_0 = 30.0
            hidden_omega_0 = 30.0
            self.linear1 = SineLayer(
                self.input_dim,
                self.bottleneck_size,
                is_first=True,
                omega_0=first_omega_0,
            )
            self.linear2 = SineLayer(
                self.bottleneck_size,
                self.bottleneck_size // 2,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            self.linear3 = SineLayer(
                self.bottleneck_size // 2,
                self.bottleneck_size // 4,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            self.linear4 = SineLayer(
                self.bottleneck_size // 4,
                self.bottleneck_size // 4,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            self.linear5 = nn.Conv1d(self.bottleneck_size // 4, 3, 1)

            with torch.no_grad():
                self.linear5.weight.uniform_(
                    -np.sqrt(6 / self.bottleneck_size) / hidden_omega_0,
                    np.sqrt(6 / self.bottleneck_size) / hidden_omega_0,
                )

        self.adain1 = AdaptiveInstanceNorm1d(self.bottleneck_size)
        self.adain2 = AdaptiveInstanceNorm1d(self.bottleneck_size // 2)
        self.adain3 = AdaptiveInstanceNorm1d(self.bottleneck_size // 4)

        self.bn1 = torch.nn.BatchNorm1d(
            self.bottleneck_size
        )  # default with Learnable Parameters
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

        self.use_SElayer = use_SElayer
        if self.use_SElayer:
            self.se1 = SELayer1D(channel=self.bottleneck_size)
            self.se2 = SELayer1D(channel=self.bottleneck_size // 2)
            self.se3 = SELayer1D(channel=self.bottleneck_size // 4)

    def forward(self, x):
        if self.use_sine:
            x = x.clone().detach().requires_grad_(True)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.linear4(x)
            x = self.linear5(x)
        else:
            if self.use_SElayer:
                x = F.relu(self.se1(self.bn1(self.adain1(self.conv1(x)))))
                x = F.relu(self.se2(self.bn2(self.adain2(self.conv2(x)))))
                x = F.relu(self.se3(self.bn3(self.adain3(self.conv3(x)))))
            else:
                x = F.relu(self.bn1(self.adain1(self.conv1(x))))
                x = F.relu(self.bn2(self.adain2(self.conv2(x))))
                x = F.relu(self.bn3(self.adain3(self.conv3(x))))
            x = self.th(self.conv4(x))
        return x


class SineLayer(nn.Module):
    """
    input:
    - x: b x (x,y) x (num_points / nb_primitives)

    output:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: int = 30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Conv1d(in_features, out_features, 1, bias=bias)
        self.adain = AdaptiveInstanceNorm1d(out_features)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        """
        input:
        - x: b x (x,y) x (num_points / nb_primitives)

        output:
        - out: b x num_dims x (num_points / nb_primitives)
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.adain(self.omega_0 * self.linear(input)))