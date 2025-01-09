from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import argparse

import trimesh

try:
    input = raw_input
except NameError:
    pass

import open3d as o3d

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import numpy as np
import tqdm
import logging

logger = logging.getLogger(__name__)

import bvh_distance_queries


def visualize_poses(poses, mesh, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=1)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box, mesh]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()

if __name__ == "__main__":
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()

    mesh_fn = "/home/wzielonka/projects/tetrahedralize/input/actor_00/object.obj"
    seed = 42

    input_mesh = trimesh.load_mesh(mesh_fn, process=False)
    if seed is not None:
        torch.manual_seed(seed)

    print(f'Number of triangles = {input_mesh.faces.shape[0]}')

    v = input_mesh.vertices

    vertices = torch.tensor(v, dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.faces.astype(np.int64),
                         dtype=torch.long,
                         device=device)

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    ##### GENERATE RAYS ####
    
    W = 2048
    H = 2048
    FL = 1000
    C = W / 2
    B = 1
    
    T = torch.tensor([[0, 0, 3]]).cuda()
    
    poses = torch.eye(4)[None].cuda()
    poses[..., :3, 3] = T
    
    poses[..., :3, 2] *= -1
    poses[..., :3, 1] *= -1
    
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    
    zs = torch.ones_like(i)
    xs = (i - C) / FL * zs
    ys = (j - C) / FL * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = (directions @ poses[:, :3, :3].transpose(-1, -2)).contiguous()

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d).contiguous()

    # visualize_poses(poses.cpu().numpy(), input_mesh)

    bvh_ray = bvh_distance_queries.BVHRayIntersection(queue_size=256)

    torch.cuda.synchronize()
    start = time.perf_counter()
    distances, closest_points, closest_faces, closest_bcs = bvh_ray(triangles, rays_o, rays_d)
    torch.cuda.synchronize()
    print(f'CUDA Ray Intersection {time.perf_counter() - start}')

    distances = distances.detach().cpu().numpy()
    mask = (distances < 10)
    closest_points = closest_points.detach().cpu().numpy().squeeze()[mask[0], ...]
    
    mask = closest_faces[(closest_faces < triangles.shape[1]) & (closest_faces >= 0)]
    centorids = triangles[0][mask, ...].mean(1).detach().cpu().numpy()
    
    ##### VISUALIZE #####

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(input_mesh.faces.astype(np.int64))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.3, 0.3])

    centorids_pcl = o3d.geometry.PointCloud()
    centorids_pcl.points = o3d.utility.Vector3dVector(centorids.reshape(-1, 3))
    centorids_pcl.paint_uniform_color([0.9, 0.3, 0.3])

    closest_points_pcl = o3d.geometry.PointCloud()
    closest_points_pcl.points = o3d.utility.Vector3dVector(closest_points.reshape(-1, 3))
    closest_points_pcl.paint_uniform_color([0.3, 0.3, 0.9])

    o3d.visualization.draw_geometries([
        mesh,
        closest_points_pcl
    ])

    ##### VISUALIZE #####
