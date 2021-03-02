# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

# differentiable point cloud rendering
import math
import torch
from cuda.p2i_op import p2i

N_VIEWS_PREDEFINED = 8


def normalize(x, dim):
    return x / torch.max(x.norm(None, dim=dim, keepdim=True), torch.tensor(1e-6, dtype=x.dtype, device=x.device))


def look_at(eyes, centers, ups):
    """look at
    Inputs:
    - eyes: float, [batch x 3], position of the observer's eye
    - centers: float, [batch x 3], where the observer is looking at (the lookat point in the above image)
    - ups: float, [batch x 3], the upper head direction of the observer

    Returns:
    - view_mat: float, [batch x 4 x 4]
    """
    zaxis = normalize(eyes - centers, dim=1)  # The negative 'forward' direction of the observer
    xaxis = normalize(torch.cross(ups, zaxis, dim=1), dim=1)  # The 'right' direction of the observer
    yaxis = torch.cross(zaxis, xaxis, dim=1)  # The rectified 'up' direction of the observer

    # constant placeholders
    zeros_pl = torch.zeros([eyes.size(0)], dtype=eyes.dtype, device=eyes.device)
    ones_pl = torch.ones([eyes.size(0)], dtype=eyes.dtype, device=eyes.device)

    translation = torch.stack(
        [
            ones_pl,
            zeros_pl,
            zeros_pl,
            -eyes[:, 0],
            zeros_pl,
            ones_pl,
            zeros_pl,
            -eyes[:, 1],
            zeros_pl,
            zeros_pl,
            ones_pl,
            -eyes[:, 2],
            zeros_pl,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
        -1,
    ).view(
        -1, 4, 4
    )  # translate coordinates so that the eyes becomes (0,0,0)

    orientation = torch.stack(
        [
            xaxis[:, 0],
            xaxis[:, 1],
            xaxis[:, 2],
            zeros_pl,
            yaxis[:, 0],
            yaxis[:, 1],
            yaxis[:, 2],
            zeros_pl,
            zaxis[:, 0],
            zaxis[:, 1],
            zaxis[:, 2],
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
        -1,
    ).view(
        -1, 4, 4
    )  # rotate the coordinates so that the above zaxis becomes (0,0,1), yaxis becomes (0,1,0), xaxis becomes (1,0,0)

    return orientation @ translation  # first translate, then orientate


def perspective(fovy, aspect, z_near, z_far):
    """perspective (right hand_no)
    Inputs:
    - fovy: float, [batch], fov angle
    - aspect: float, [batch], aspect ratio
    - z_near, z_far: float, [batch], the z-clipping distances

    Returns:
    - proj_mat: float, [batch x 4 x 4]
    """
    tan_half_fovy = torch.tan(fovy / 2.0)
    zeros_pl = torch.zeros_like(fovy)
    ones_pl = torch.ones_like(fovy)

    k1 = -(z_far + z_near) / (z_far - z_near)
    k2 = -2.0 * z_far * z_near / (z_far - z_near)
    return torch.stack(
        [
            1.0 / aspect / tan_half_fovy,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            1.0 / tan_half_fovy,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            k1,
            k2,
            zeros_pl,
            zeros_pl,
            -ones_pl,
            zeros_pl,
        ],
        -1,
    ).view(-1, 4, 4)


def orthorgonal(scalex, scaley, z_near, z_far):
    zeros_pl = torch.zeros_like(z_near)
    ones_pl = torch.ones_like(z_near)

    k1 = -2.0 / (z_far - z_near)
    k2 = (z_far + z_near) / (z_far - z_near)
    return torch.stack(
        [
            scalex,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            scaley,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            k1,
            k2,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
        -1,
    ).view(-1, 4, 4)


def transform(matrix, points):
    """
    Inputs:
    - matrix: float, [npoints x 4 x 4]
    - points: float, [npoints x 3]

    Outputs:
    - transformed_points: float, [npoints x 3]
    """
    out = torch.cat([points, torch.ones_like(points[:, [0]], device=points.device)], dim=1).view(points.size(0), 4, 1)
    out = matrix @ out
    out = out[:, :3, 0] / out[:, [3], 0]
    return out


class ComputeDepthMaps(torch.nn.Module):
    def __init__(self, projection: str = "orthorgonal", eyepos_scale: float = 1.0, image_size: int = 256):
        super().__init__()

        self.image_size = image_size
        self.eyes_pos_list = [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
        self.num_views = len(self.eyes_pos_list)
        assert projection in {"perspective", "orthorgonal"}
        if projection == "perspective":
            self.projection_matrix = perspective(
                fovy=torch.tensor([math.pi / 4], dtype=torch.float32),
                aspect=torch.tensor([1.0], dtype=torch.float32),
                z_near=torch.tensor([0.1], dtype=torch.float32),
                z_far=torch.tensor([10.0], dtype=torch.float32),
            )
        else:
            self.projection_matrix = orthorgonal(
                scalex=torch.tensor([1.5], dtype=torch.float32),
                scaley=torch.tensor([1.5], dtype=torch.float32),
                z_near=torch.tensor([0.1], dtype=torch.float32),
                z_far=torch.tensor([10.0], dtype=torch.float32),
            )

        self.pre_matrix_list = []
        for i in range(self.num_views):
            _view_matrix = look_at(
                eyes=torch.tensor([self.eyes_pos_list[i]], dtype=torch.float32) * eyepos_scale,  # can multiply 0.8 if the eye is too close?
                centers=torch.tensor([[0, 0, 0]], dtype=torch.float32),
                ups=torch.tensor([[0, 0, 1]], dtype=torch.float32),
            )

            self.register_buffer("_pre_matrix", self.projection_matrix @ _view_matrix)
            self.pre_matrix_list.append(self._pre_matrix)

    def forward(self, data, view_id=0, radius_list=[10.0]):
        if view_id >= self.num_views:
            return None

        _batch_size = data.size(0)
        _num_points = data.size(1)
        _matrix = self.pre_matrix_list[view_id].expand(_batch_size * _num_points, 4, 4).to(data.device)
        _background = torch.zeros(_batch_size, 1, self.image_size, self.image_size, dtype=data.dtype, device=data.device)
        _batch_inds = torch.arange(0, _batch_size, dtype=torch.int32, device=data.device)
        _batch_inds = _batch_inds.unsqueeze(1).expand(_batch_size, _num_points).reshape(-1)

        pcds = data.view(-1, 3)  # [bs* num_points, 3]
        trans_pos = transform(_matrix, pcds)
        pos_xs, pos_ys, pos_zs = trans_pos.split(dim=1, split_size=1)
        pos_ijs = torch.cat([-pos_ys, pos_xs], dim=1)  # negate pos_ys because images row indices are from top to bottom
        point_features = 1.0 - (pos_zs - pos_zs.min()) / (pos_zs.max() - pos_zs.min())  # npoints x 1, a one-channel point feature

        # depth_maps: [bs, 1, 256, 256]
        # depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())
        for radius in radius_list:
            if radius == radius_list[0]:
                depth_maps = p2i(
                    pos_ijs,
                    point_features,
                    _batch_inds,
                    _background,
                    kernel_radius=radius_list[0],
                    kernel_kind_str="cos",
                    reduce="max",
                )
            else:
                _depth_maps = p2i(
                    pos_ijs,
                    point_features,
                    _batch_inds,
                    _background,
                    kernel_radius=radius,
                    kernel_kind_str="cos",
                    reduce="max",
                )
                depth_maps = torch.cat((depth_maps, _depth_maps), dim=1)
        return depth_maps


if __name__ == "__main__":
    import datasets.data_loaders
    import utils.misc
    from configs.base_config import cfg
    import torch
    import os

    if not os.path.exists("__temp__"):
        os.mkdir("__temp__")
    import time
    import torchvision

    dataset_loader = datasets.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(datasets.data_loaders.DatasetSubset.TEST),
        batch_size=1,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=datasets.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False,
    )

    # give the positions of the observers' eyes
    # todo: cannonical views?
    eyes_grid = [
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
    ]

    # construct the module
    compute_depth_maps = ComputeDepthMaps(projection="perspective", eyepos_scale=0.01, image_size=cfg.RENDER.IMG_SIZE).float()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        if model_idx % 100 == 0:

            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            for k, v in data.items():
                data[k] = utils.misc.var_or_cuda(v)

            for i in range(8):
                depth_map = compute_depth_maps(data["gtcloud"], view_id=i, radius_list=cfg.RENDER.RADIUS_LIST)  # [bs, len(radius_list), 256, 256]
                for j in range(len(cfg.RENDER.RADIUS_LIST)):
                    torchvision.utils.save_image(depth_map[:, j, :, :], f"__temp__/depth_maps_p_{model_idx}_v{i}_r{j}.jpg", pad_value=1)

            start = time.time()
            depth_map = compute_depth_maps(data["gtcloud"], view_id=0, radius_list=cfg.RENDER.RADIUS_LIST)
            stop = time.time()
            print(f"{stop - start} seconds")
