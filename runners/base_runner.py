# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

# python3.6
"""Contains the base class for runner.
This runner can be used for training and with multi-threads.
"""

import os
import yaml
import torch
import pprint
import numpy as np
from time import time
from copy import deepcopy
from runners.misc import AverageMeter
from datasets.data_loaders import data_init
import utils.misc as um
import utils.visualizer as uv
from utils.model_init import generator_init

class BaseRunner(object):
    """Defines the base runner class."""

    def __init__(self, config, logger):
        self._name = self.__class__.__name__
        self._config = deepcopy(config)
        self.logger = logger
        self.work_dir = self.config.DIR.out_path
        os.makedirs(self.work_dir, exist_ok=True)

        self.logger.info("Running Configuration:")
        config_str = pprint.pformat(self.config)
        self.logger.info("\n" + config_str)
        with open(os.path.join(self.work_dir, "config.yaml"), "w") as f:
            f.write(yaml.dump(vars(self.config)))

        self.gpu_ids = um.gpu_init(self.config)

        self.train_loader = None
        self.val_loader = None
        self.train_writer = None
        self.val_writer = None

        self.init_epoch = 0
        self.best_metrics = None

        self.models = None
        self.optimizers = None
        self.lr_schedulers = None

        self.loss = {}
        self.losses = None
        self.test_losses = None
        self.test_metrics = None
        self.ptcloud = None

        self.metrics = None
        self.model_idx = None
        self.taxonomy_id = None
        self.model_id = None

        self.start_time = 0
        self.end_time = 0

        self.batch_idx = 0
        self.epoch_idx = 0
        self.epoch_start_time = 0
        self.epoch_end_time = 0

        self.train_time = AverageMeter()
        self.val_time = AverageMeter()

        self.build_writer()
        self.build_dataset()
        self.build_models()
        self.data_parallel()
        self.models_load()

    @property
    def name(self):
        """Returns the name of the runner."""
        return self._name

    @property
    def config(self):
        """Returns the configuration of the runner."""
        return self._config

    def build_dataset(self):
        """Builds train/val dataset."""
        self.train_loader, self.val_loader = data_init(self.config)
        self.logger.info(f"Finish building dataset.")

    def build_models(self):
        """Builds models, optimizers, and learning rate schedulers."""
        self.models, self.optimizers, self.lr_schedulers = generator_init(self.config)

    def data_parallel(self):
        """Sets `self.model` as `torch.nn.parallel.DistributedDataParallel`."""
        self.models = torch.nn.DataParallel(self.models.to(self.gpu_ids[0]), device_ids=self.gpu_ids)

    def models_load(self):
        """Load models"""
        self.init_epoch, self.best_metrics = um.model_load(self.config, self.models)

    def models_save(self):
        """Save models"""
        self.best_metrics = um.checkpoint_save(self.config, self.epoch_idx, self.metrics, self.best_metrics, self.models)

    def build_writer(self):
        """Builds additional controllers besides LRScheduler."""
        self.train_writer, self.val_writer = um.writer_init(self.config)

    def set_mode(self, mode):
        """Sets the `train/val` mode for all models."""
        self.mode = mode
        if mode == "train" or mode is True:
            self.models.train()
        elif mode in ["val", "test", "eval"] or mode is False:
            self.models.eval()
        else:
            raise ValueError(f"Invalid model mode `{mode}`!")

    def train_step(self, data):
        """Executes one training step."""
        raise NotImplementedError("Should be implemented in derived class.")

    def save_item_train_info(self):
        # keep the train loss

        n_itr = (self.epoch_idx - 1) * self.n_batches + self.batch_idx
        if self.batch_idx % self.config.TRAIN.log_freq == 0:
            for k, v in self.loss.items():
                self.train_writer.add_scalar("Loss/Batch/" + k, v.item(), n_itr)
            self.logger.info(
                "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) Losses = %s"
                % (
                    self.epoch_idx,
                    self.config.TRAIN.n_epochs,
                    self.batch_idx + 1,
                    self.n_batches,
                    self.train_time.val(),
                    ["%.4f" % l for l in self.losses.val()],
                )
            )

    def train(self):
        """Training function."""
        self.set_mode("train")
        self.logger.info(f"Start training.")
        self.epoch_start_time = time()

        for items in enumerate(self.train_loader):
            self.batch_idx, _ = items
            self.n_batches = len(self.train_loader)
            train_start_time = time()
            # run the completion network
            self.train_step(items)
            self.train_time.update(time() - train_start_time)
            self.save_item_train_info()

        self.train_finish()

    def val_step(self, data):
        """Executes one validate step."""
        raise NotImplementedError("Should be implemented in derived class.")

    def save_item_val_info(self, data):
        # keep the val loss

        self.test_metrics.update(self.metrics)
        if self.taxonomy_id not in self.category_metrics:
            self.category_metrics[self.taxonomy_id] = AverageMeter(um.Metrics.names())
        self.category_metrics[self.taxonomy_id].update(self.metrics)

        if self.model_idx % self.config.TRAIN.log_freq == 0:
            self.logger.info(
                "Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s"
                % (
                    self.model_idx + 1,
                    self.n_batches,
                    self.taxonomy_id,
                    self.model_id,
                    ["%.4f" % l for l in self.test_losses.val()],
                    ["%.4f" % m for m in self.metrics],
                )
            )
        self.inference(data)

    def val(self):
        """Validation function."""
        self.category_metrics = {}
        self.set_mode("val")
        self.logger.info(f"Start validating.")

        for items in enumerate(self.val_loader):
            self.model_idx, (taxonomy_id, _, model_id, data) = items
            self.taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            self.model_id = model_id[0]
            self.n_batches = len(self.val_loader)
            val_start_time = time()
            # run the completion network
            self.val_step(items)
            self.val_time.update(time() - val_start_time)
            self.save_item_val_info(data)

        self.metrics = um.Metrics(self.config.TEST.metric_name, self.test_metrics.avg())
        self.val_finish()

    def train_finish(self):
        """Finishes runner."""
        self.lr_schedulers.step()
        self.epoch_end_time = time()

        for i in range(len(self.losses.items)):
            self.train_writer.add_scalar("Loss/Epoch/" + self.losses.items[i], self.losses.avg(i), self.epoch_idx)

        self.logger.info(
            "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s"
            % (
                self.epoch_idx,
                self.config.TRAIN.n_epochs,
                self.epoch_end_time - self.epoch_start_time,
                ["%.4f" % l for l in self.losses.avg()],
            )
        )

    def val_finish(self):
        """Finishes runner."""
        uv.print_table(
            self.config,
            self.epoch_idx,
            self.test_metrics,
            self.category_metrics,
            self.val_writer,
            self.test_losses,
        )
        self.models_save()

    def build_train_loss(self):
        """build_train_loss"""
        raise NotImplementedError("Should be implemented in derived class.")

    def build_val_loss(self):
        """build_val_loss"""
        raise NotImplementedError("Should be implemented in derived class.")

    def inference(self, data):
        for k, v in data.items():
            data[k] = v.float().to(self.gpu_ids[0])

        if self.model_idx % self.config.TEST.infer_freq == 0:

            if self.config.TEST.mode == "default":
                uv.tensorflow_save_image(
                    refine_ptcloud=self.ptcloud,
                    data=data,
                    test_writer=self.val_writer,
                    model_idx=self.model_idx,
                    epoch_idx=self.epoch_idx,
                )

            elif self.config.TEST.mode == "vis":
                os.makedirs(os.path.join(self.config.DIR.logs, "plots", self.taxonomy_id), exist_ok=True)
                plot_path = os.path.join(self.config.DIR.logs, "plots", self.taxonomy_id, "%s.png" % self.model_idx)
                print("save image", plot_path)
                uv.plot_pcd_three_views(
                    plot_path,
                    [
                        data["partial_cloud"].squeeze().cpu(),
                        self.ptcloud.squeeze().cpu(),
                        data["gtcloud"].squeeze().cpu(),
                    ],
                    ["input", "output", "ground truth"],
                    "CD %.4f  EMD %.4f F-score %.4f" % (self.metrics[1], self.metrics[2], self.metrics[0]),
                    [5, 0.5, 0.5],
                )

            elif self.config.TEST.mode == "render":
                os.makedirs(os.path.join(self.config.DIR.logs, "plots", self.taxonomy_id), exist_ok=True)
                uv.save_depth_map(
                    cfg=self.config,
                    refine_ptcloud=self.ptcloud,
                    data=data,
                    taxonomy_id=self.taxonomy_id,
                    model_idx=self.model_idx,
                )

            elif self.config.TEST.mode == "kitti":
                output_folder = os.path.join(self.config.DIR.out_path, "benchmark", self.taxonomy_id)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                output_file_path = os.path.join(output_folder, "%s.h5" % self.model_idx)
                uv.IO.put(output_file_path, self.ptcloud.squeeze().cpu().numpy())
                self.logger.info(
                    "Test[%d/%d] Taxonomy = %s Sample = %s File = %s" % (self.model_idx + 1, self.n_batches, self.taxonomy_id, self.model_idx, output_file_path)
                )

    def runner(self):
        """Runner"""
        self.start_time = time()
        for epoch_idx in range(self.init_epoch + 1, self.config.TRAIN.n_epochs + 1):
            self.epoch_idx = epoch_idx
            self.build_train_loss()
            self.train()
            self.build_val_loss()
            with torch.no_grad():
                self.val()
        self.end_time = time()
        self.logger.info("runner time: %3f" % (self.end_time - self.start_time))
        self.train_writer.close()
        self.val_writer.close()

    def test(self):
        """test"""
        assert self.init_epoch != 0
        with torch.no_grad():
            self.build_val_loss()
            self.start_time = time()
            self.epoch_idx = -1
            self.val()
            self.end_time = time()
            self.logger.info("test time: %3f" % (self.end_time - self.start_time))
            self.train_writer.close()
            self.val_writer.close()
