# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import random
from enum import Enum, unique
import numpy as np
import torch.utils.data.dataset
from tqdm import tqdm
import datasets.data_transforms
from datasets.io import IO

logger = logging.getLogger()


def data_init(cfg):
    """
    input:
        cfg: EasyDict

    outputs:
        train_data_loader: DataLoader
        val_data_loader: DataLoader
    """
    # Set up data loader
    train_dataset_loader = DATASET_LOADER_MAPPING[cfg.DATASET.train_dataset](cfg)
    test_dataset_loader = DATASET_LOADER_MAPPING[cfg.DATASET.test_dataset](cfg)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.batch_size,
        num_workers=cfg.CONST.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    if cfg.DATASET.test_dataset == "Completion3D":
        val_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset_loader.get_dataset(DatasetSubset.VAL),
            batch_size=1,
            num_workers=cfg.CONST.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )
    else:
        val_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset_loader.get_dataset(DatasetSubset.TEST),
            batch_size=1,
            num_workers=cfg.CONST.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )
    if cfg.GAN.use_cgan:
        # will be used in define_G function and creating discriminator
        cfg.DATASET.num_classes = len(train_dataset_loader.dataset_categories)
        if cfg.DATASET.train_dataset == "Completion3D":
            cfg.DATASET.num_classes -= 1
        logger.debug("update config NUM_CLASSES: %d." % cfg.DATASET.num_classes)
    return (train_data_loader, val_data_loader)


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    labels = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        labels.append(sample[1])
        model_ids.append(sample[2])
        _data = sample[3]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, labels, model_ids, data


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if "n_renderings" in self.options:
            rand_idx = (
                random.randint(0, self.options["n_renderings"] - 1)
                if self.options["shuffle"]
                else 0
            )

        for ri in self.options["required_items"]:
            file_path = sample["%s_path" % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]

            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample["taxonomy_id"], sample["label"], sample["model_id"], data


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.shapenet.category_file_path) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = (
            self.cfg.DATASETS.shapenet.n_renderings
            if subset == DatasetSubset.TRAIN
            else 1
        )
        file_list = self._get_file_list(
            self.cfg, self._get_subset(subset), n_renderings
        )
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset(
            {
                "required_items": ["partial_cloud", "gtcloud"],
                "shuffle": subset == DatasetSubset.TRAIN,
            },
            file_list,
            transforms,
        )

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return datasets.data_transforms.Compose(
                [
                    {
                        "callback": "RandomSamplePoints",
                        "parameters": {"n_points": 3000},
                        "objects": ["partial_cloud"],
                    },
                    {
                        "callback": "RandomSamplePoints",
                        "parameters": {"n_points": cfg.DATASET.n_outpoints},
                        "objects": ["gtcloud"],
                    },
                    {
                        "callback": "RandomMirrorPoints",
                        "objects": ["partial_cloud", "gtcloud"],
                    },
                    {"callback": "ToTensor", "objects": ["partial_cloud", "gtcloud"]},
                ]
            )
        else:
            return datasets.data_transforms.Compose(
                [
                    {
                        "callback": "RandomSamplePoints",
                        "parameters": {"n_points": 3000},
                        "objects": ["partial_cloud"],
                    },
                    {
                        "callback": "RandomSamplePoints",
                        "parameters": {"n_points": cfg.DATASET.n_outpoints},
                        "objects": ["gtcloud"],
                    },
                    {"callback": "ToTensor", "objects": ["partial_cloud", "gtcloud"]},
                ]
            )

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return "train"
        elif subset == DatasetSubset.VAL:
            return "val"
        else:
            return "test"

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        for label, dc in enumerate(self.dataset_categories):
            logger.info(
                "Collecting files of Taxonomy [ID=%s, Name=%s]"
                % (dc["taxonomy_id"], dc["taxonomy_name"])
            )
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                # original impletementation from GRNet,
                # will randomly select 1 view out of 8 view point cloud in partial_cloud_path for each model (see line 66)

                if cfg.DATASETS.shapenet.version == "GRnet":
                    file_list.append(
                        {
                            "taxonomy_id": dc["taxonomy_id"],
                            "label": label,
                            "model_id": s,
                            "partial_cloud_path": [
                                cfg.DATASETS.shapenet.partial_points_path
                                % (subset, dc["taxonomy_id"], s, i)
                                for i in range(n_renderings)
                            ],
                            "gtcloud_path": cfg.DATASETS.shapenet.complete_points_path
                            % (subset, dc["taxonomy_id"], s),
                        }
                    )

                else:
                    # we can use the completed dataset from shapenet as following
                    # add 8 views point cloud of each model in the file_list
                    for i in range(n_renderings):
                        file_list.append(
                            {
                                "taxonomy_id": dc["taxonomy_id"],
                                "label": label,
                                "model_id": s + str(i),
                                "partial_cloud_path": cfg.DATASETS.shapenet.partial_points_path
                                % (subset, dc["taxonomy_id"], s, i),
                                "gtcloud_path": cfg.DATASETS.shapenet.complete_points_path
                                % (subset, dc["taxonomy_id"], s),
                            }
                        )

        logger.info(
            "Complete collecting files of the dataset. Total files: %d" % len(file_list)
        )
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [
            dc for dc in self.dataset_categories if dc["taxonomy_id"] == "02958343"
        ]


class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.completion3d.category_file_path) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = (
            ["partial_cloud"]
            if subset == DatasetSubset.TEST
            else ["partial_cloud", "gtcloud"]
        )

        return Dataset(
            {
                "required_items": required_items,
                "shuffle": subset == DatasetSubset.TRAIN,
            },
            file_list,
            transforms,
        )

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return datasets.data_transforms.Compose(
                [
                    {
                        "callback": "RandomSamplePoints",
                        "parameters": {"n_points": cfg.CONST.n_input_points},
                        "objects": ["partial_cloud"],
                    },
                    {
                        "callback": "RandomMirrorPoints",
                        "objects": ["partial_cloud", "gtcloud"],
                    },
                    {"callback": "ToTensor", "objects": ["partial_cloud", "gtcloud"]},
                ]
            )
        else:
            return datasets.data_transforms.Compose(
                [
                    {
                        "callback": "RandomSamplePoints",
                        "parameters": {"n_points": cfg.CONST.n_input_points},
                        "objects": ["partial_cloud"],
                    },
                    {"callback": "ToTensor", "objects": ["partial_cloud", "gtcloud"]},
                ]
            )

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return "train"
        elif subset == DatasetSubset.VAL:
            return "val"
        else:
            return "test"

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []
        label = 0
        for dc in self.dataset_categories:
            logger.info(
                "Collecting files of Taxonomy [ID=%s, Name=%s]"
                % (dc["taxonomy_id"], dc["taxonomy_name"])
            )
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append(
                    {
                        "taxonomy_id": dc["taxonomy_id"],
                        "label": label,
                        "model_id": s,
                        "partial_cloud_path": cfg.DATASETS.completion3d.partial_points_path
                        % (subset, dc["taxonomy_id"], s),
                        "gtcloud_path": cfg.DATASETS.completion3d.complete_points_path
                        % (subset, dc["taxonomy_id"], s),
                    }
                )
            if dc["taxonomy_id"] != "all":
                label += 1

        logger.info(
            "Complete collecting files of the dataset. Total files: %d" % len(file_list)
        )
        return file_list


class KittiDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.kitti.category_file_path) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ["partial_cloud", "bounding_box"]

        return Dataset(
            {"required_items": required_items, "shuffle": False}, file_list, transforms
        )

    def _get_transforms(self, cfg, subset):
        return datasets.data_transforms.Compose(
            [
                {
                    "callback": "NormalizeObjectPose",
                    "parameters": {
                        "input_keys": {
                            "ptcloud": "partial_cloud",
                            "bbox": "bounding_box",
                        }
                    },
                    "objects": ["partial_cloud", "bounding_box"],
                },
                {
                    "callback": "RandomSamplePoints",
                    "parameters": {"n_points": cfg.CONST.n_input_points},
                    "objects": ["partial_cloud"],
                },
                {"callback": "ToTensor", "objects": ["partial_cloud", "bounding_box"]},
            ]
        )

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return "train"
        elif subset == DatasetSubset.VAL:
            return "val"
        else:
            return "test"

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []
        label = 0
        for dc in self.dataset_categories:
            logger.info(
                "Collecting files of Taxonomy [ID=%s, Name=%s]"
                % (dc["taxonomy_id"], dc["taxonomy_name"])
            )
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append(
                    {
                        "taxonomy_id": dc["taxonomy_id"],
                        "label": label,
                        "model_id": s,
                        "partial_cloud_path": cfg.DATASETS.kitti.partial_points_path
                        % s,
                        "bounding_box_path": cfg.DATASETS.kitti.bounding_box_file_path
                        % s,
                    }
                )

        logger.info(
            "Complete collecting files of the dataset. Total files: %d" % len(file_list)
        )
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    "Completion3D": Completion3DDataLoader,
    "ShapeNet": ShapeNetDataLoader,
    "ShapeNetCars": ShapeNetCarsDataLoader,
    "KITTI": KittiDataLoader,
}  # yapf: disable
