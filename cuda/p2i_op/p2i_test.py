import os
import torch
from torch.autograd import gradcheck, grad

import time

from . import p2i


def test1():
    points = torch.zeros(1, 2, dtype=torch.float64, device="cuda")
    point_features = torch.ones(1, 3, dtype=torch.float64, device="cuda")
    batch_inds = torch.arange(1, dtype=torch.int32, device="cuda")
    background = torch.zeros(1, 3, 8, 8, dtype=torch.float64, device="cuda")

    out = p2i(points, point_features, batch_inds, background, 2, "cos", "sum")
    print(out)

    out = p2i(points, point_features, batch_inds, background, 2, "cos", "max")
    print(out)


def test2():
    for i in range(10):
        print(i)
        points = torch.randn(2, 2, dtype=torch.float64, device="cuda")
        point_features = torch.randn(2, 3, dtype=torch.float64, device="cuda")
        batch_inds = torch.zeros(2, dtype=torch.int32, device="cuda")
        background = torch.randn(1, 3, 8, 8, dtype=torch.float64, device="cuda")

        points.requires_grad = True
        point_features.requires_grad = True
        background.requires_grad = True
        gradcheck(p2i, inputs=(points, point_features, batch_inds, background, 2, "cos", "sum"))
        gradcheck(p2i, inputs=(points, point_features, batch_inds, background, 2, "cos", "max"))


if __name__ == "__main__":
    test2()
