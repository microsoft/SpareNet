# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

import os
import torch
import logging
from time import time
import utils.misc as um
import cuda.emd.emd_module as emd
from cuda.chamfer_distance import ChamferDistance, ChamferDistanceMean
from runners.misc import AverageMeter
from runners.base_runner import BaseRunner


class sparenetRunner(BaseRunner):
    """Define the SpareNet runner class"""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.losses = AverageMeter(["CoarseLoss", "RefineLoss"])
        self.test_losses = AverageMeter(["CoarseLoss", "RefineLoss"])
        self.test_metrics = AverageMeter(um.Metrics.names())
        self.chamfer_dist = None
        self.chamfer_dist_mean = None
        self.emd_dist = None

    def build_models(self):
        super().build_models()

    def build_train_loss(self):
        # Set up loss functions
        self.chamfer_dist = torch.nn.DataParallel(ChamferDistance().to(self.gpu_ids[0]), device_ids=self.gpu_ids)
        self.chamfer_dist_mean = torch.nn.DataParallel(ChamferDistanceMean().to(self.gpu_ids[0]), device_ids=self.gpu_ids)
        self.emd_dist = torch.nn.DataParallel(emd.emdModule().to(self.gpu_ids[0]), device_ids=self.gpu_ids)

    def build_val_loss(self):
        # Set up loss functions
        self.chamfer_dist = ChamferDistance().cuda()
        self.chamfer_dist_mean = ChamferDistanceMean().cuda()
        self.emd_dist = emd.emdModule().cuda()

    def train_step(self, items):
        _, (_, _, _, data) = items
        for k, v in data.items():
            data[k] = v.float().to(self.gpu_ids[0])

        _loss, _, _, _, refine_loss, coarse_loss = self.completion(data)
        self.models.zero_grad()
        _loss.backward()
        self.optimizers.step()

        self.loss["coarse_loss"] = coarse_loss * 1000
        self.loss["refine_loss"] = refine_loss * 1000
        self.loss["rec_loss"] = _loss
        self.losses.update([coarse_loss.item() * 1000, refine_loss.item() * 1000])

    def val_step(self, items):
        _, (_, _, _, data) = items
        for k, v in data.items():
            data[k] = um.var_or_cuda(v)

        _, refine_ptcloud, _, _, refine_loss, coarse_loss = self.completion(data)
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
            middle_ptcloud: b x npoints2 x num_dims
            coarse_ptcloud: b x npoints2 x num_dims
            refine_loss: float32
            coarse_loss: float32
        """
        (coarse_ptcloud, middle_ptcloud, refine_ptcloud, expansion_penalty) = self.models(data)

        if self.config.NETWORK.metric == "chamfer":
            coarse_loss = self.chamfer_dist_mean(coarse_ptcloud, data["gtcloud"]).mean()
            middle_loss = self.chamfer_dist_mean(middle_ptcloud, data["gtcloud"]).mean()
            refine_loss = self.chamfer_dist_mean(refine_ptcloud, data["gtcloud"]).mean()

        elif self.config.NETWORK.metric == "emd":
            emd_coarse, _ = self.emd_dist(coarse_ptcloud, data["gtcloud"], eps=0.005, iters=50)
            emd_middle, _ = self.emd_dist(middle_ptcloud, data["gtcloud"], eps=0.005, iters=50)
            emd_refine, _ = self.emd_dist(refine_ptcloud, data["gtcloud"], eps=0.005, iters=50)
            coarse_loss = torch.sqrt(emd_coarse).mean(1).mean()
            refine_loss = torch.sqrt(emd_refine).mean(1).mean()
            middle_loss = torch.sqrt(emd_middle).mean(1).mean()

        else:
            raise Exception("unknown training metric")

        _loss = coarse_loss + middle_loss + refine_loss + expansion_penalty.mean() * 0.1

        if self.config.NETWORK.use_consist_loss:
            dist1, _ = self.chamfer_dist(refine_ptcloud, data["gtcloud"])
            cd_input2fine = torch.mean(dist1).mean()
            _loss += cd_input2fine * 0.5

        return _loss, refine_ptcloud, middle_ptcloud, coarse_ptcloud, refine_loss, coarse_loss