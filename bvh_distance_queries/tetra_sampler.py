import sys

from typing import Tuple, NewType

import torch
import torch.nn as nn
import torch.autograd as autograd

import bvh_distance_queries_cuda

Tensor = NewType('Tensor', torch.Tensor)


class SamplerFunction(autograd.Function):
    SORT_POINTS_BY_MORTON = True

    @staticmethod
    def forward(ctx,
                intersecting_tetras: Tensor,
                topology: Tensor,
                triangles: Tensor,
                ro: Tensor,
                rd: Tensor,
                ray_mask: Tensor,
                step_size: float,
                max_samples: int
            ):

        outputs = bvh_distance_queries_cuda.sampling(
            intersecting_tetras, topology, triangles, ro, rd, ray_mask, step_size, max_samples
        )

        ctx.save_for_backward(triangles, *outputs)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6]

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


class Sampler(nn.Module):
    def __init__(self) -> None:
        super(Sampler, self).__init__()

    @torch.no_grad()
    def forward(
            self,
            intersecting_tetras: Tensor,
            topology: Tensor,
            triangles: Tensor,
            ro: Tensor,
            rd: Tensor,
            ray_mask: Tensor,
            step_size: float,
            max_samples: int
        ):

        return SamplerFunction.apply(intersecting_tetras, topology, triangles, ro, rd, ray_mask, step_size, max_samples)
