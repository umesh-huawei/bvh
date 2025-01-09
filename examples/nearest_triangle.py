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

if __name__ == "__main__":
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()

    mesh_fn = "/home/wzielonka/projects/tetrahedralize/input/actor_00/object.obj"
    num_query_points = 10000
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

    min_vals, _ = torch.min(vertices, dim=0, keepdim=True)
    max_vals, _ = torch.max(vertices, dim=0, keepdim=True)

    query_points = torch.rand([1, num_query_points, 3], dtype=torch.float32,device=device) * (max_vals - min_vals) + min_vals
    query_points_np = query_points.detach().cpu().numpy().squeeze(axis=0).astype(np.float32).reshape(num_query_points, 3)

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    m = bvh_distance_queries.BVH()

    torch.cuda.synchronize()
    start = time.perf_counter()
    distances, closest_points, closest_faces, closest_bcs = m(triangles, query_points)
    torch.cuda.synchronize()
    print(f'CUDA Elapsed time {time.perf_counter() - start}')
    
    closest_points = closest_points.detach().cpu().numpy().squeeze()
    
    ##### VISUALIZE #####

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(input_mesh.faces.astype(np.int64))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.3, 0.3])

    query_pcl = o3d.geometry.PointCloud()
    query_pcl.points = o3d.utility.Vector3dVector(query_points.detach().cpu().numpy().squeeze(axis=0).reshape(-1, 3))
    query_pcl.paint_uniform_color([0.9, 0.3, 0.3])

    closest_points_pcl = o3d.geometry.PointCloud()
    closest_points_pcl.points = o3d.utility.Vector3dVector(closest_points.reshape(-1, 3))
    closest_points_pcl.paint_uniform_color([0.3, 0.3, 0.9])

    o3d.visualization.draw_geometries([
        mesh,
        query_pcl,
        closest_points_pcl
    ])
    
    ##### VISUALIZE #####
