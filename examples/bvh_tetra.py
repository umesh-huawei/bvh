import os
from tqdm import tqdm

import meshio
import trimesh
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import bvh_distance_queries


class TetraSampler(nn.Module):
    def __init__(self, path) -> None:
        super(TetraSampler, self).__init__()
        mesh = meshio.read(path)

        self.tetras = torch.from_numpy(mesh.cells_dict['tetra']).cuda()

        tris = mesh.cells_dict['triangle']
        mesh = trimesh.Trimesh(vertices=mesh.points, faces=tris, process=False)
        trimesh.repair.fix_normals(mesh)

        self.triangles = torch.from_numpy(mesh.faces).cuda()
        self.points = torch.from_numpy(mesh.vertices).float().cuda()

        self.A = self.tetras[:, 0]
        self.B = self.tetras[:, 1]
        self.C = self.tetras[:, 2]
        self.D = self.tetras[:, 3]
        
        v0 = torch.stack([self.A, self.B, self.C], dim=1)
        v1 = torch.stack([self.A, self.B, self.D], dim=1)
        v2 = torch.stack([self.A, self.C, self.D], dim=1)
        v3 = torch.stack([self.B, self.C, self.D], dim=1)
        
        self.trimesh = mesh;
        self.ABCD = torch.stack([v0, v1, v2, v3], dim=1)
        self.tetra_faces = self.ABCD.view(-1, 3)
        self.triangle_to_tetra = torch.repeat_interleave(torch.arange(self.tetras.shape[0]), 4).unsqueeze(1).cuda()
        self.topology = self.build_topology()
        self.tatra_color = np.array([np.random.choice(range(256), size=3) / 256 for _ in range(self.topology.shape[0])])
    
    def build_topology(self):
        N = self.tetra_faces.shape[0]
        sorted_faces = torch.sort(self.tetra_faces, dim=1)[0].cpu().numpy()
        shared_triangle = {}
        for i in tqdm(range(N)):
            tet_id = int(i / 4)
            vert_id = i % 4
            x = sorted_faces[i]
            key = hash(x.tostring())
            payload = (tet_id, vert_id)
            if key not in shared_triangle:
                shared_triangle[key] = [payload]
            else:
                shared_triangle[key].append(payload) 
            
        topology = np.ones_like(self.ABCD.cpu().numpy())[..., 0] * -1
        
        items = list(shared_triangle.items())
        
        for l in range(0, len(items)):
            v = items[l][1]
            if len(v) == 2:
                tet_id_0, vert_id_0 = v[0]
                tet_id_1, vert_id_1 = v[1]

                topology[tet_id_0, vert_id_0] = tet_id_1
                topology[tet_id_1, vert_id_1] = tet_id_0

        return torch.from_numpy(topology).float().cuda()
    
    def draw(self, geometries=[]):
        N = self.tetra_faces.shape[0]
        
        v = self.points.detach().cpu().numpy()
        f = self.tetra_faces.detach().cpu().numpy()
        t = self.tetras.detach().cpu().numpy()
        
        tetra = o3d.geometry.TetraMesh()
        tetra.vertices = o3d.utility.Vector3dVector(v)
        tetra.tetras = o3d.utility.Vector4iVector(t.astype(np.int64))
        tetra.paint_uniform_color([0, 0, 1.0])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int64))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0, 0, 1.0])
        
        query_pcl = o3d.geometry.PointCloud()
        query_pcl.points = o3d.utility.Vector3dVector(v)
        query_pcl.paint_uniform_color([0.9, 0.3, 0.3])

        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        for geo in [tetra] + geometries:
            viewer.add_geometry(geo)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        viewer.run()
        viewer.destroy_window()
    
    def forward(self, cage_vertices, ro, rd):
        B = ro.shape[0]
        triangles = cage_vertices[:, self.ABCD] # batch size for now only 1, number of rays will vary
        bvh_ray = bvh_distance_queries.BVHRayIntersection(queue_size=128)
        distances, closest_points, closest_faces = bvh_ray(triangles.view(B, -1, 3, 3), ro, rd)
         
        # Filter the hit triangle MAX 5 meters
        mask = distances < 5
        distances = distances[mask]
        closest_faces = closest_faces[mask]
        ro = closest_points[mask] # Set sampling origin at intersection position
        rd = rd[mask]
        closest_tetras = self.triangle_to_tetra[closest_faces].squeeze(1)
        
        # Sanity check. All egde tetras shoud have at most one -1 as neighbour
        check = self.topology[closest_tetras].cpu().numpy()
        check = np.sort(check)
        all_negative = np.sum(check[:, 0] == -1)
        assert(all_negative == closest_tetras.shape[0])
    
        sampling = bvh_distance_queries.Sampler()
        
        sampling_length = 0.05  # in meters
        max_samples = 128
        step_size = sampling_length / max_samples

        ray_indices, tetra_indices, bary_coords, t_start, t_end, deformed_positions = sampling(
            closest_tetras.unsqueeze(0).contiguous().int(),
            self.topology.unsqueeze(0).contiguous(),
            triangles.contiguous(),
            ro.unsqueeze(0).contiguous(),
            rd.unsqueeze(0).contiguous(),
            distances.unsqueeze(0).contiguous(),
            step_size, 
            max_samples
        )

        return ray_indices, tetra_indices, bary_coords, t_start, t_end, deformed_positions
