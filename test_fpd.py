import open3d as o3d
import argparse
import logging
import torch
import os
import numpy as np
from Frechet.FPD import calculate_fpd
logger = logging.getLogger()

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return torch.from_numpy(np.array(pcd.points)).float()


def save_pcd(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def set_logger(filename):
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



parser = argparse.ArgumentParser()
parser.add_argument('--plot_freq', type=int, default=1)
parser.add_argument('--save_pcd', action='store_true',default=False)
parser.add_argument('--log_dir', default='/path/to/save/logs')
parser.add_argument('--list_path', default='Frechet/test.list')
parser.add_argument('--data_dir', default='/path/to/test/dataset/pcds')
parser.add_argument('--fake_dir', default='/path/to/methods/pcds',
                            help='/path/to/results/shapenet_fc/pcds/')
parser.add_argument('--num_points', type=int, default=16384, help='number of points: 2048 or 8192')



opt = parser.parse_args()
os.makedirs(opt.log_dir, exist_ok=True)

set_logger(os.path.join(opt.log_dir, "log.txt"))
logger.info("save into dir: %s" % opt.log_dir)


with open(opt.list_path) as file:
    model_list = file.read().splitlines()


# Testing loop
n_samples = len(model_list)
logger.info("n_samples %s"% n_samples)


taxonomy2label = {
    '02691156':0, 
    '02933112':1, 
    '02958343':2,
    '03001627':3,
    '03636649':4,
    '04256520':5,
    '04379243':6,
    '04530566':7
}


label2taxonomy = {
    0:'02691156', 
    1:'02933112', 
    2:'02958343',
    3:'03001627',
    4:'03636649',
    5:'04256520',
    6:'04379243',
    7:'04530566'
}

fpd_values= []

# test for each category 
for batch_idx in range(0,8):
    fake_pointclouds =  torch.Tensor([]).cuda()
    real_pointclouds =  torch.Tensor([]).cuda()
    # prepare the batch data
    for idx, model_id in enumerate(model_list):
        taxonomy_id, model_id_real = model_id.split('/')
        if taxonomy2label[taxonomy_id] == batch_idx:
            fake = torch.zeros((1, opt.num_points, 3), device='cuda')
            gt = torch.zeros((1, opt.num_points, 3), device='cuda')
            pcd = o3d.io.read_point_cloud(os.path.join(opt.fake_dir,  '%s.pcd' % model_id))
            fake[0, :, :] = torch.from_numpy(np.array(pcd.points))
            pcd = o3d.io.read_point_cloud(os.path.join(opt.data_dir, 'complete', '%s.pcd' % model_id))
            gt[0, :, :] = torch.from_numpy(np.array(pcd.points))
            fake_pointclouds = torch.cat((fake_pointclouds, fake), dim=0)
            real_pointclouds = torch.cat((real_pointclouds, gt), dim=0)

    # 150 data samples per batch 
    fpd = calculate_fpd(fake_pointclouds, real_pointclouds, statistic_save_path=None, batch_size=30, dims=1808, device= fake_pointclouds.device)
    fpd_values.append(fpd)
    logger.info("[ %s  category] Frechet Pointcloud Distance <<< %f >>>"% (label2taxonomy[batch_idx], fpd))

    del fake_pointclouds
    del real_pointclouds

logger.info("avg FPD for 8 categories: %f" % np.mean(fpd_values))

# test the all test data 


fake_pointclouds =  torch.Tensor([]).cuda()
real_pointclouds =  torch.Tensor([]).cuda()
# prepare the batch data
for idx, model_id in enumerate(model_list):
    taxonomy_id, model_id_real = model_id.split('/')
    fake = torch.zeros((1, opt.num_points, 3), device='cuda')
    gt = torch.zeros((1, opt.num_points, 3), device='cuda')
    pcd = o3d.io.read_point_cloud(os.path.join(opt.fake_dir,  '%s.pcd' % model_id))
    fake[0, :, :] = torch.from_numpy(np.array(pcd.points))
    pcd = o3d.io.read_point_cloud(os.path.join(opt.data_dir, 'complete', '%s.pcd' % model_id))
    gt[0, :, :] = torch.from_numpy(np.array(pcd.points))
    fake_pointclouds = torch.cat((fake_pointclouds, fake), dim=0)
    real_pointclouds = torch.cat((real_pointclouds, gt), dim=0)

# 150 data samples per batch 
fpd = calculate_fpd(fake_pointclouds, real_pointclouds, statistic_save_path=None, batch_size=30, dims=1808, device= fake_pointclouds.device)
logger.info("[ all category] Frechet Pointcloud Distance <<< %f >>>"%  fpd)




