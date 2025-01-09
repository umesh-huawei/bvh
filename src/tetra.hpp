#ifndef TETRA_H
#define TETRA_H

#include "defs.hpp"
#include "double_vec_ops.h"
#include "helper_math.h"
#include "triangle.hpp"

#include "math_utils.hpp"
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

template <typename T>
struct Tetra
{
  Triangle<T> v0{};
  Triangle<T> v1{};
  Triangle<T> v2{};
  Triangle<T> v3{};

  __host__ __device__ Tetra() {}
  __host__ __device__ Tetra(Triangle<T> _v0, Triangle<T> _v1, Triangle<T> _v2, Triangle<T> _v3) : v0(_v0), v1(_v1), v2(_v2), v3(_v3){};
  __host__ __device__ Tetra(const Triangle<T> &_v0, const Triangle<T> &_v1, const Triangle<T> &_v2, const Triangle<T> &_v3) : v0(_v0), v1(_v1), v2(_v2), v3(_v3){};
};

template <typename T>
using TetraPtr = Tetra<T> *;

#endif // TETRA_H
