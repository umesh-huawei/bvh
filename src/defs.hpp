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

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

template <typename T>
using vec4 = typename std::conditional<std::is_same<T, float>::value, float4, double4>::type;

template <typename T>
using vec3 = typename std::conditional<std::is_same<T, float>::value, float3, double3>::type;

template <typename T>
using vec2 = typename std::conditional<std::is_same<T, float>::value, float2, double2>::type;

float4 make_float4(double4 vec) {
    return make_float4(vec.x, vec.y, vec.z, vec.w);
}

float4 make_float4(double x, double y, double z, double w) {
    return make_float4(x, y, z, w);
}

double4 make_double4(float4 vec) {
    return make_double4(vec.x, vec.y, vec.z, vec.w);
}

double4 make_double4(float x, float y, float z, float w) {
    return make_double4(x, y, z, w);
}

float3 make_float3(double3 vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

float3 make_float3(double x, double y, double z) {
    return make_float3(x, y, z);
}

double3 make_double3(float3 vec) {
    return make_double3(vec.x, vec.y, vec.z);
}

double3 make_double3(float x, float y, float z) {
    return make_double3(x, y, z);
}

float2 make_float2(double2 vec) {
    return make_float2(vec.x, vec.y);
}

float2 make_float2(double x, double y) {
    return make_float2(x, y);
}

double2 make_double2(float2 vec) {
    return make_double2(vec.x, vec.y);
}

double2 make_double2(float x, float y) {
    return make_double2(x, y);
}

template <typename T>
__host__ __device__
vec4<T> make_vec4(T x, T y, T z, T w) {
}

template <typename T>
__host__ __device__
vec3<T> make_vec3(T x, T y, T z) {
}

template <typename T>
__host__ __device__
vec2<T> make_vec2(T x, T y) {
}

template <>
__host__ __device__
vec4<float> make_vec4(float x, float y, float z, float w) {
    return make_float4(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), static_cast<float>(w));
}

template <>
__host__ __device__
vec4<double> make_vec4(double x, double y, double z, double w) {
    return make_double4(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z), static_cast<double>(w));
}

template <>
__host__ __device__
vec3<float> make_vec3(float x, float y, float z) {
    return make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

template <>
__host__ __device__
vec3<double> make_vec3(double x, double y, double z) {
    return make_double3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
}

template <>
__host__ __device__
vec2<float> make_vec2(float x, float y) {
    return make_float2(static_cast<float>(x), static_cast<float>(y));
}

template <>
__host__ __device__
vec2<double> make_vec2(double x, double y) {
    return make_double2(static_cast<double>(x), static_cast<double>(y));
}

#endif // ifndef DEFINITIONS_H
