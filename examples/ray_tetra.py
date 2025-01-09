from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import argparse
from bvh_tetra import TetraSampler

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

if __name__ == "__main__":
    device = torch.device('cuda')
    
    ##### GENERATE RAYS ####
    
    W = 3000
    H = 3000
    FL = 1000
    C = W / 2
    B = 1
    
    T = torch.tensor([[0, 1.5, 3]]).cuda()
    
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

    #########################
    
    tetraBVH = TetraSampler("/home/wzielonka/projects/tetrahedralize/output/actor_02/cage.mesh")      
    cage_vertices = tetraBVH.points[None].expand(B, -1, -1)

    torch.cuda.synchronize()
    start = time.perf_counter()
    ray_indices, tetra_indices, bary_coords, t_start, t_end, deformed_positions = tetraBVH(cage_vertices, rays_o, rays_d)
    torch.cuda.synchronize()
    print(f'CUDA Tet Ray Sampling #N_RAYS {rays_o.shape[1]} Time {time.perf_counter() - start}')

    #### PACK SAMPLES ####
    
    used_cells = ray_indices >= 0
    
    deformed_positions = deformed_positions[used_cells]
    ray_indices = ray_indices[used_cells]
    tetra_indices = tetra_indices[used_cells]
    bary_coords = bary_coords[used_cells]
    t_start = t_start[used_cells]
    t_end = t_end[used_cells]

    #### PROJECT POINTS TO CANONICAL  ####
    
    coords = torch.stack([cage_vertices[:, tetraBVH.A], cage_vertices[:, tetraBVH.B], cage_vertices[:, tetraBVH.C], cage_vertices[:, tetraBVH.D]], dim=2).transpose(3, 2)
    tetras = coords[:, tetra_indices, ...]
    
    vertices = torch.einsum("bijk,bik->bij", tetras, bary_coords[None])
    
    trimesh.points.PointCloud(vertices=vertices[0].cpu().numpy()).export("canonical.ply")
    trimesh.points.PointCloud(vertices=deformed_positions.cpu().numpy()).export("deformed.ply")
    
    ########################

    idx = tetra_indices.cpu().numpy().reshape(-1)
    bary = bary_coords.cpu().numpy()
   
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tetraBVH.trimesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tetraBVH.trimesh.faces)
    
    pos_pcl = o3d.geometry.PointCloud()
    pos_pcl.points = o3d.utility.Vector3dVector(deformed_positions.cpu().numpy().reshape(-1, 3))
    pos_pcl.colors = o3d.utility.Vector3dVector(tetraBVH.tatra_color[idx])

    o3d.visualization.draw_geometries([
       pos_pcl,
       o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ])
