# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

import os
import cv2
import h5py
import json
import open3d
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.p2i_utils import ComputeDepthMaps
from cuda.chamfer_distance import ChamferDistance, ChamferDistanceMean


def get_ptcloud_img(ptcloud):
    """ Point cloud visualization via matplotlib """
    fig = plt.figure(figsize=(3, 3))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable="box")
    ax.axis("off")
    # ax.axis('scaled')
    ax.view_init(30, -45)

    # max, min = np.max(ptcloud), np.min(ptcloud)
    # ax.set_xbound(min, max)
    # ax.set_ybound(min, max)
    # ax.set_zbound(min, max)
    ax.set_xlim((-0.3, 0.3))
    ax.set_ylim((-0.3, 0.3))
    ax.set_zlim((-0.3, 0.3))
    ax.scatter(x, y, z, zdir="z", c=x, cmap="jet")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")

    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


def plot_pcd_three_views(
    filename,
    pcds,
    titles,
    suptitle="",
    sizes=None,
    cmap="Reds",
    zdir="y",
    xlim=(-0.3, 0.3),
    ylim=(-0.3, 0.3),
    zlim=(-0.3, 0.3),
):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    elev = 30
    for i in range(3):  # plot three views
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection="3d")
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def print_table(
    cfg,
    epoch_idx,
    test_metrics,
    category_metrics,
    test_writer,
    test_losses,
):
    log_table = {"epoch": epoch_idx}
    print("============================ TEST RESULTS ============================")
    print("epoch", epoch_idx)
    print("Taxonomy", end="\t")
    print("#Sample", end="\t")
    for metric in test_metrics.items:
        print(metric, end="\t")
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end="\t")
        print(category_metrics[taxonomy_id].count(0), end="\t")
        for value in category_metrics[taxonomy_id].avg():
            print("%.4f" % value, end="\t")
        print()

        for i, m in enumerate(category_metrics[taxonomy_id].items):
            log_table[str(taxonomy_id) + "_" + m] = "%.6f" % category_metrics[taxonomy_id].avg(i)

    print("Overall", end="\t\t\t")
    for value in test_metrics.avg():
        print("%.4f" % value, end="\t")
    print("\n")

    for i, m in enumerate(test_metrics.items):
        log_table["overall" + "_" + m] = "%.6f" % test_metrics.avg(i)

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar("Loss/Epoch/Sparse", test_losses.avg(0), epoch_idx)
        test_writer.add_scalar("Loss/Epoch/Dense", test_losses.avg(1), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar("Metric/%s" % metric, test_metrics.avg(i), epoch_idx)

    with open(os.path.join(cfg.DIR.logs, "test.txt"), "a") as f:
        f.write("json_stats: " + json.dumps(log_table) + "\n")


def tensorflow_save_image(
    refine_ptcloud,
    data,
    test_writer,
    model_idx,
    epoch_idx,
):
    partical_ptcloud = data["partial_cloud"].squeeze().cpu()
    partical_ptcloud_img = get_ptcloud_img(partical_ptcloud)
    test_writer.add_image("Model%02d/ParticalReconstruction" % model_idx, np.transpose(partical_ptcloud_img, (2, 0, 1)), 0)
    refine_ptcloud = refine_ptcloud.squeeze().cpu()
    refine_ptcloud_img = get_ptcloud_img(refine_ptcloud)
    test_writer.add_image("Model%02d/DenseReconstruction" % model_idx, np.transpose(refine_ptcloud_img, (2, 0, 1)), epoch_idx)
    gt_ptcloud = data["gtcloud"].squeeze().cpu()
    gt_ptcloud_img = get_ptcloud_img(gt_ptcloud)
    test_writer.add_image("Model%02d/GroundTruth" % model_idx, np.transpose(gt_ptcloud_img, (2, 0, 1)), 1)


def save_depth_map(
    cfg,
    refine_ptcloud,
    data,
    taxonomy_id,
    model_idx,
):
    compute_depth_maps = ComputeDepthMaps(
        projection=cfg.RENDER.projection,
        eyepos_scale=cfg.RENDER.eyepos,
        image_size=cfg.RENDER.img_size,
    ).float()
    for j in range(len(compute_depth_maps.eyes_pos_list)):  # 8 views // 4 = 2 view
        gen_render_imgs = compute_depth_maps(refine_ptcloud, j, radius_list=[7.0])  # Manually use radius 7.0
        plot_path = os.path.join(cfg.DIR.logs, "plots", taxonomy_id, str(model_idx), str(j) + "2.png")
        print("save image", plot_path)
        torchvision.utils.save_image(gen_render_imgs, plot_path, pad_value=1)

        gen_render_imgs = compute_depth_maps(data["partial_cloud"], j, radius_list=[7.0])  # Manually use radius 7.0
        plot_path = os.path.join(cfg.DIR.logs, "plots", taxonomy_id, str(model_idx), str(j) + "1.png")
        print("save image", plot_path)
        torchvision.utils.save_image(gen_render_imgs, plot_path, pad_value=1)

        gen_render_imgs = compute_depth_maps(data["gtcloud"], j, radius_list=[7.0])  # Manually use radius 7.0
        plot_path = os.path.join(cfg.DIR.logs, "plots", taxonomy_id, str(model_idx), str(j) + "3.png")
        print("save image", plot_path)
        torchvision.utils.save_image(gen_render_imgs, plot_path, pad_value=1)


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
