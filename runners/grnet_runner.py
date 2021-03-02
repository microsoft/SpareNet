# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

import os
import torch
import logging
from time import time
import utils.misc as um
import cuda.emd.emd_module as emd
from cuda.chamfer_distance import ChamferDistanceMean
from runners.misc import AverageMeter
from runners.base_runner import BaseRunner


class grnetRunner(BaseRunner):
    """Define the AtlasNet runner class"""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.losses = AverageMeter(["CoarseLoss", "RefineLoss"])
        self.test_losses = AverageMeter(["CoarseLoss", "RefineLoss"])
        self.test_metrics = AverageMeter(um.Metrics.names())
        self.chamfer_dist_mean = None
        self.emd_dist = None

    def build_models(self):
        super().build_models()

    def build_train_loss(self):
        # Set up loss functions
        self.chamfer_dist_mean = torch.nn.DataParallel(ChamferDistanceMean().to(self.gpu_ids[0]), device_ids=self.gpu_ids)
        self.emd_dist = torch.nn.DataParallel(emd.emdModule().to(self.gpu_ids[0]), device_ids=self.gpu_ids)

    def build_val_loss(self):
        # Set up loss functions
        self.chamfer_dist_mean = ChamferDistanceMean().cuda()
        self.emd_dist = emd.emdModule().cuda()

    def train_step(self, items):

        # prepare the data and label
        _, (_, _, _, data) = items
        for k, v in data.items():
            data[k] = v.float().to(self.gpu_ids[0])

        # run the completion network
        _loss, _, _, refine_loss, coarse_loss = self.completion(data)
        self.models.zero_grad()
        _loss.backward()
        self.optimizers.step()

        self.loss["coarse_loss"] = coarse_loss * 1000
        self.loss["refine_loss"] = refine_loss * 1000
        self.loss["rec_loss"] = _loss
        self.losses.update([coarse_loss * 1000, refine_loss * 1000])

    def val_step(self, items):
        _, (_, _, _, data) = items

        # prepare the data and label
        for k, v in data.items():
            data[k] = um.var_or_cuda(v)

        # run the completion network
        _, _, refine_ptcloud, coarse_loss, refine_loss = self.completion(data)
        self.test_losses.update([coarse_loss.item() * 1000, refine_loss.item() * 1000])
        self.metrics = um.Metrics.get(refine_ptcloud, data["gtcloud"])
        self.ptcloud = refine_ptcloud

    def completion(self, data):
        """
        inputs:
            cfg: EasyDict
            data: tensor
                -partical_cloud: b x npoints1 x num_dims
                -gtcloud: b x npoints2 x num_dims

        outputs:
            _loss: float32
            refine_ptcloud: b x npoints2 x num_dims
            coarse_ptcloud: b x npoints2 x num_dims
            refine_loss: float32
            coarse_loss: float32
        """
        # there is the middle pcd between coarse pcd and refined pcd
        (
            coarse_ptcloud,
            refine_ptcloud,
        ) = self.models(data)

        # here we can choose the way to compute the loss
        coarse_loss = self.chamfer_dist_mean(coarse_ptcloud, data["gtcloud"]).mean()

        if self.config.NETWORK.metric == "chamfer":
            refine_loss = self.chamfer_dist_mean(refine_ptcloud, data["gtcloud"]).mean()

        elif self.config.NETWORK.metric == "emd":
            emd_refine, _ = self.emd_dist(refine_ptcloud, data["gtcloud"], eps=0.005, iters=50)
            refine_loss = torch.sqrt(emd_refine).mean(1).mean()

        else:
            raise Exception("unknown training metric")

        # loss is the sum of coarse loss, meddle loss, refined loss and expansion penalty loss,(and consistent loss)
        _loss = coarse_loss + refine_loss

        return _loss, coarse_ptcloud, refine_ptcloud, coarse_loss, refine_loss
