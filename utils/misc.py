# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

import os
import torch
import open3d
import logging
from tensorboardX import SummaryWriter
import cuda.emd.emd_module as emd
from cuda.chamfer_distance import ChamferDistanceMean

logger = logging.getLogger()

##### type conversion #####
def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def gpu_init(cfg):
    """
    input:
        cfg: EasyDict

    output:
        gup_ids: list
    """
    # Set up folders for checkpoints
    if not os.path.exists(cfg.DIR.checkpoints):
        os.makedirs(cfg.DIR.checkpoints)
    # GPU setup
    torch.backends.cudnn.benchmark = True
    gup_ids = [int(x) for x in cfg.CONST.device.split(",")]
    return list(range(len(gup_ids)))


def writer_init(cfg):
    """
    input:
        cfg: EasyDict

    outputs:
        train_writer: SummaryWriter
        val_writer: SummaryWriter
    """
    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.logs, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.logs, "test"))
    return train_writer, val_writer


def model_load(cfg, net_G):
    """
    load model

    inputs:
        cfg: EasyDict
        net_G: torch.nn.module

    outputs:
        init_epoch: int
        best_metrics: dic
    """

    init_epoch = 0
    best_metrics = None
    # Load pretrained model if exists
    if cfg.CONST.weights:
        logger.info("Recovering from %s ..." % (cfg.CONST.weights))
        checkpoint = torch.load(cfg.CONST.weights)
        best_metrics = Metrics(cfg.TEST.metric_name, checkpoint["best_metrics"])
        init_epoch = checkpoint["epoch_index"]
        net_G.load_state_dict(checkpoint["net_G"])  # change into net_G!!
        logger.info("Recover complete. Current epoch = #%d; best metrics = %s." % (init_epoch, best_metrics))
    return init_epoch, best_metrics


def checkpoint_save(cfg, epoch_idx, metrics, best_metrics, net_G):
    """
    save the model

    inputs:
        cfg: EasyDict
        epoch_idx: int
        metrics: dic
        best_metrics: dic
        net_G: torch.nn.module

    outputs:
        best_metrics: dic
    """
    # save tbe best model
    if epoch_idx % cfg.TRAIN.save_freq == 0 or metrics.better_than(best_metrics):
        file_name = "ckpt-best.pth" if metrics.better_than(best_metrics) else "ckpt-epoch-%03d.pth" % epoch_idx
        output_path = os.path.join(cfg.DIR.checkpoints, file_name)
        # save the epoch and metrics and net_G
        state = {
            "epoch_index": epoch_idx,
            "best_metrics": metrics.state_dict(),
            "net_G": net_G.state_dict(),
        }
        torch.save(state, output_path)

        logger.info("Saved checkpoint to %s ..." % output_path)
        if metrics.better_than(best_metrics):
            best_metrics = metrics
    return best_metrics


def set_logger(filename):
    """s
    set logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s: - %(message)s")

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


class Metrics(object):
    ITEMS = [
        {
            "name": "F-Score",
            "enabled": True,
            "eval_func": "cls._get_f_score",
            "is_greater_better": True,
            "init_value": 0,
        },
        {
            "name": "ChamferDistance",
            "enabled": True,
            "eval_func": "cls._get_chamfer_distance",
            "eval_object": ChamferDistanceMean(),
            "is_greater_better": False,
            "init_value": 32767,
        },
        {
            "name": "EMD",
            "enabled": True,
            "eval_func": "cls._get_emd",
            "eval_object": emd.emdModule(),
            "is_greater_better": False,
            "init_value": 32767,
        },
    ]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item["eval_func"])
            _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i["enabled"]]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i["name"] for i in _items]

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        pred = cls._get_open3d_ptcloud(pred)
        gt = cls._get_open3d_ptcloud(gt)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))
        return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distance(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]["eval_object"]
        return chamfer_distance(pred, gt).item() * 1000

    @classmethod
    def _get_emd(cls, pred, gt):
        EMD = cls.ITEMS[2]["eval_object"]
        dist, _ = EMD(pred, gt, eps=0.005, iters=50)  # for val
        # dist, _ = EMD(pred, gt, 0.002, 10000) # final test ?
        emd = torch.sqrt(dist).mean(1).mean()
        return emd.item() * 100

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item["init_value"] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == "dict":
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item["name"]
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logger.warn("Ignore Metric[Name=%s] due to disability." % k)
                    continue
                self._values[metric_indexes[k]] = v
        elif type(values).__name__ == "list":
            self._values = values
        else:
            raise Exception("Unsupported value type: %s" % type(values))

    def state_dict(self):
        _dict = {}
        for i in range(len(self._items)):
            item = self._items[i]["name"]
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item["name"] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception("Invalid metric name to compare.")

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric["is_greater_better"] else _value < other_value
