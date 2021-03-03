# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from io import BytesIO

import cv2
import h5py
import numpy as np
import open3d

from configs.base_config import cfg


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".png", ".jpg"]:
            return cls._read_img(file_path)
        elif file_extension in [".npy"]:
            return cls._read_npy(file_path)
        elif file_extension in [".pcd"]:
            return cls._read_pcd(file_path)
        elif file_extension in [".h5"]:
            return cls._read_h5(file_path)
        elif file_extension in [".txt"]:
            return cls._read_txt(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def put(cls, file_path, file_content):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".pcd"]:
            return cls._write_pcd(file_path, file_content)
        elif file_extension in [".h5"]:
            return cls._write_h5(file_path, file_content)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def _read_img(cls, file_path):
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED) / 255.0

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # NOTE: Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        return np.array(pc.points)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, "r")
        # Avoid overflow while gridding
        return f["data"][()] * 0.9

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _write_pcd(cls, file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc)

    @classmethod
    def _write_h5(cls, file_path, file_content):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data", data=file_content)
