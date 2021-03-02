import os

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

__all__ = ["p2i"]

module_path = os.path.dirname(os.path.abspath(__file__))
ext = load(
    "ext",
    sources=[
        os.path.join(module_path, "ext.cpp"),
        os.path.join(module_path, "p2i_sum.cu"),
        os.path.join(module_path, "p2i_max.cu"),
    ],
    extra_cuda_cflags=["--expt-extended-lambda", "-O3"],
)


class P2ISumFunction(Function):
    @staticmethod
    def forward(ctx, points, point_features, batch_inds, background, kernel_kind, kernel_radius):
        ctx.save_for_backward(points, point_features, batch_inds)
        ctx.kernel_kind = kernel_kind
        ctx.kernel_radius = kernel_radius

        out = ext.p2i_sum_forward_gpu(
            points.contiguous(),
            point_features.contiguous(),
            batch_inds.contiguous(),
            background.contiguous(),
            kernel_kind,
            kernel_radius,
        )

        return out

    @staticmethod
    def backward(ctx, out_grad):
        points, point_features, batch_inds = ctx.saved_tensors
        kernel_kind = ctx.kernel_kind
        kernel_radius = ctx.kernel_radius

        points_grad, point_features_grad = ext.p2i_sum_backward_gpu(
            out_grad.contiguous(),
            points.contiguous(),
            point_features.contiguous(),
            batch_inds.contiguous(),
            kernel_kind,
            kernel_radius,
        )

        background_grad = out_grad
        return points_grad, point_features_grad, None, background_grad, None, None


class P2IMaxFunction(Function):
    @staticmethod
    def forward(ctx, points, point_features, batch_inds, background, kernel_kind, kernel_radius):

        out, out_point_ids = ext.p2i_max_forward_gpu(
            points.contiguous(),
            point_features.contiguous(),
            batch_inds.contiguous(),
            background.contiguous(),
            kernel_kind,
            kernel_radius,
        )

        ctx.save_for_backward(points, point_features, out_point_ids)
        ctx.kernel_kind = kernel_kind
        ctx.kernel_radius = kernel_radius

        return out

    @staticmethod
    def backward(ctx, out_grad):
        points, point_features, out_point_ids = ctx.saved_tensors
        kernel_kind = ctx.kernel_kind
        kernel_radius = ctx.kernel_radius

        points_grad, point_features_grad, background_grad = ext.p2i_max_backward_gpu(
            out_grad.contiguous(),
            out_point_ids,
            points.contiguous(),
            point_features.contiguous(),
            kernel_kind,
            kernel_radius,
        )

        return points_grad, point_features_grad, None, background_grad, None, None


_kernel_kind_dict = {"cos": 0}


def p2i(points, point_features, batch_inds, background, kernel_radius, kernel_kind_str="cos", reduce="sum"):
    r"""p2i

    Paint point cloud features on to a 2D feature map.

    inputs:
      - points: float, [npoints x 2], (+/-1, +/-1) represents the image corners
      - point_features: float, [npoints x channels]
      - batch_inds: int32, [npoints]
      - background: float, [batch x channels x out_h x out_w]
      - kernel_radius: float
      - kernel_kind_str: str, {'cos'}
      - reduce: str, {'sum', 'max'}
    returns:
      - output: float, [batch x channels x out_h x out_w]
    """
    kernel_kind = _kernel_kind_dict[kernel_kind_str]
    out_h, out_w = background.shape[2:]
    points = (
        (points + 1)
        / 2
        * torch.tensor([out_h - 1, out_w - 1], dtype=points.dtype, device=points.device).view(1, 2)
    )

    if reduce == "sum":
        return P2ISumFunction.apply(
            points, point_features, batch_inds, background, kernel_kind, kernel_radius
        )
    elif reduce == "max":
        return P2IMaxFunction.apply(
            points, point_features, batch_inds, background, kernel_kind, kernel_radius
        )
    raise RuntimeError(f"Invalid reduce value: {reduce}")

custom_fun = P2ISumFunction.apply
