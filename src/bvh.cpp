/*
 * Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
 * holder of all proprietary rights on this computer program.
 * You can only use this computer program if you have closed
 * a license agreement with MPG or you get the right to use the computer
 * program from someone who is authorized to grant you that right.
 * Any use of the computer program without a valid license is prohibited and
 * liable to prosecution.
 *
 * Copyright©2019 Max-Planck-Gesellschaft zur Förderung
 * der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
 * for Intelligent Systems. All rights reserved.
 *
 * @author Vasileios Choutas
 * Contact: vassilis.choutas@tuebingen.mpg.de
 * Contact: ps-license@tuebingen.mpg.de
 *
 */

#include <iostream>
#include <vector>
#include <limits>

#include <torch/extension.h>

void bvh_distance_queries_kernel(
    const torch::Tensor &triangles,
    const torch::Tensor &points,
    torch::Tensor *distances,
    torch::Tensor *closest_points,
    torch::Tensor *closest_faces,
    torch::Tensor *closest_bcs,
    int queue_size = 128,
    bool sort_points_by_morton = true);

void bvh_ray_intersection_kernel(
    const torch::Tensor &triangles,
    const torch::Tensor &ro,
    const torch::Tensor &rd,
    torch::Tensor *distances,
    torch::Tensor *closest_points,
    torch::Tensor *closest_faces,
//     torch::Tensor *closest_bcs,
//     torch::Tensor *hit_points,
//     torch::Tensor *hit_faces,
//     torch::Tensor *n_hits,
//     const int max_hits,
    int queue_size = 128);

void ray_sampler_kernel(
    const torch::Tensor &intersecting_tetras,
    const torch::Tensor &topology,
    const torch::Tensor &triangles,
    const torch::Tensor &ro,
    const torch::Tensor &rd,
    const torch::Tensor &startts,
    float step_size,
    int max_samples,
    torch::Tensor *chunks,
    torch::Tensor *ray_indices,
    torch::Tensor *tetra_indices,
    torch::Tensor *bary_coords,
    torch::Tensor *t_starts,
    torch::Tensor *t_ends,
    torch::Tensor *deformed_positions);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
        CHECK_CUDA(x); \
        CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bvh_distance_queries(
    torch::Tensor triangles,
    torch::Tensor points,
    int queue_size = 128,
    bool sort_points_by_morton = true)
{
        CHECK_INPUT(triangles);
        CHECK_INPUT(points);

        auto options = torch::TensorOptions()
                           .dtype(triangles.dtype())
                           .layout(triangles.layout())
                           .device(triangles.device());

        torch::Tensor distances = torch::full({triangles.size(0), points.size(1)}, -1, options);
        torch::Tensor closest_points = torch::full({triangles.size(0), points.size(1), 3}, -1, options);
        torch::Tensor closest_bcs = torch::full({triangles.size(0), points.size(1), 3}, 0, options);
        torch::Tensor closest_faces = torch::full({triangles.size(0), points.size(1)}, -1, torch::TensorOptions().dtype(torch::kLong).layout(triangles.layout()).device(triangles.device()));

        bvh_distance_queries_kernel(
            triangles,
            points,
            &distances,
            &closest_points,
            &closest_faces,
            &closest_bcs,
            queue_size,
            sort_points_by_morton);

        return {distances, closest_points, closest_faces, closest_bcs};
}

std::vector<torch::Tensor> bvh_ray_intersection(
    torch::Tensor triangles,
    torch::Tensor ro, torch::Tensor rd,
    int queue_size = 128)
{
        CHECK_INPUT(triangles);
        CHECK_INPUT(ro);
        CHECK_INPUT(rd);

        auto options = torch::TensorOptions()
                           .dtype(triangles.dtype())
                           .layout(triangles.layout())
                           .device(triangles.device());

        int max_intersections = 1;

        torch::Tensor distances = torch::full({triangles.size(0), ro.size(1)}, -1, options);
        torch::Tensor closest_points = torch::full({triangles.size(0), ro.size(1), 3}, -1, options);
        // torch::Tensor closest_bcs = torch::full({triangles.size(0), ro.size(1), 3}, 0, options);
        torch::Tensor closest_faces = torch::full({triangles.size(0), ro.size(1)}, -1, torch::TensorOptions().dtype(torch::kLong).layout(triangles.layout()).device(triangles.device()));
        // torch::Tensor n_hits = torch::full({triangles.size(0), ro.size(1)}, 0, torch::TensorOptions().dtype(torch::kLong).layout(triangles.layout()).device(triangles.device()));
        // torch::Tensor hit_faces = torch::full({triangles.size(0), ro.size(1), max_intersections}, 0, torch::TensorOptions().dtype(torch::kLong).layout(triangles.layout()).device(triangles.device()));
        // torch::Tensor hit_points = torch::full({triangles.size(0), ro.size(1), max_intersections, 3}, -1, options);

        bvh_ray_intersection_kernel(
            triangles,
            ro,
            rd,
            &distances,
            &closest_points,
            &closest_faces,
        //     &closest_bcs,
        //     &hit_points,
        //     &hit_faces,
        //     &n_hits,
        //     max_intersections,
            queue_size);

        return {distances, closest_points, closest_faces};
}

std::vector<torch::Tensor> sampling(
    torch::Tensor intersecting_tetras,
    torch::Tensor topology,
    torch::Tensor triangles,
    torch::Tensor ro,
    torch::Tensor rd,
    torch::Tensor startts,
    float step_size,
    int max_samples)
{

        CHECK_INPUT(triangles);
        CHECK_INPUT(ro);
        CHECK_INPUT(rd);
        CHECK_INPUT(startts);

        auto options = torch::TensorOptions()
                           .dtype(ro.dtype())
                           .layout(ro.layout())
                           .device(ro.device());

        int batch = ro.size(0);
        int n_rays = ro.size(1);
        int n_samples = n_rays * max_samples;

        torch::Tensor chunks = torch::full({batch, n_rays}, 0, torch::TensorOptions().dtype(torch::kLong).layout(ro.layout()).device(ro.device()));
        torch::Tensor bary_coords = torch::full({batch, n_samples, 4}, -1, options);
        torch::Tensor deformed_positions = torch::full({batch, n_samples, 3}, -1, options); // Defined outside the volume of interests
        torch::Tensor t_starts = torch::full({batch, n_samples}, -1, options);
        torch::Tensor t_ends = torch::full({batch, n_samples}, -1, options);
        torch::Tensor ray_indices = torch::full({batch, n_samples}, -1, torch::TensorOptions().dtype(torch::kLong).layout(ro.layout()).device(ro.device()));
        torch::Tensor tetra_indices = torch::full({batch, n_samples}, -1, torch::TensorOptions().dtype(torch::kLong).layout(ro.layout()).device(ro.device()));

        ray_sampler_kernel(
            intersecting_tetras,
            topology,
            triangles,
            ro,
            rd,
            startts,
            step_size,
            max_samples,
            &chunks,
            &ray_indices,
            &tetra_indices,
            &bary_coords,
            &t_starts,
            &t_ends,
            &deformed_positions);

        int size = chunks.sum().item<int>();
        torch::Tensor tmp = chunks.to(torch::kCPU);
        std::vector<long> chunks_cpu(tmp.data_ptr<long>(), tmp.data_ptr<long>() + tmp.numel());
        std::vector<long> packed(size, 0);

        int i = 0;
        int current_index = 0;
        while (current_index < size) {
            // int chunk_size = chunks_cpu.index({0, i}).item().toInt();
            int chunk_size = chunks_cpu[i];
            for (int j = 0; j < chunk_size; ++j) {
                packed[current_index + j] = i;
            }
            current_index += chunk_size;
            ++i;
        }

        torch::Tensor packed_info = torch::from_blob(packed.data(), {size}, torch::TensorOptions().dtype(torch::kLong));
        packed_info = packed_info.to(torch::kCUDA);

        return {ray_indices, tetra_indices, bary_coords, t_starts, t_ends, deformed_positions, packed_info};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
        m.def("distance_queries", &bvh_distance_queries, "distance queries", py::arg("triangles"), py::arg("points"), py::arg("queue_size") = 128, py::arg("sort_points_by_morton") = true);
        m.def("ray_queries", &bvh_ray_intersection, "ray triangle intersection", py::arg("triangles"), py::arg("ro"), py::arg("rd"), py::arg("queue_size") = 128);
        m.def("sampling", &sampling, "sampling", py::arg("intersecting_tetras"), py::arg("topology"), py::arg("triangles"), py::arg("ro"), py::arg("rd"), py::arg("startts"), py::arg("step_size"), py::arg("max_samples"));
}
