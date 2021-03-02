#pragma once

#ifndef __CUDACC__
#include <algorithm>
#include <atomic>
using std::max;
using std::min;
#endif

#include "common.h"

namespace haya_ext {

template <typename T> XDEVICE T atomic_add(T *addr, T v) {
#ifdef __CUDACC__
  return atomicAdd(addr, v);
#else
  return *addr += v;
#endif
}

#if defined(__CUDACC__) && __CUDA_ARCH__ < 600 // old cuda
__forceinline__ __device__ double atomic_add(double *addr, double v) {
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(v + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename T> XDEVICE T atomic_cas(T *addr, T compare, T val) {
#ifdef __CUDACC__
  return atomicCAS(addr, compare, val);
#else
  std::atomic<T> this_val(*addr);
  this_val.compare_exchange_weak(compare, val);
  return *addr = this_val.load();
#endif
}

template <typename T> XDEVICE T atomic_exch(T *addr, T v) {
#ifdef __CUDACC__
  return atomicExch(addr, v);
#else
  T rd = *addr;
  *addr = v;
  return rd;
#endif
}

#ifdef __CUDACC__
__forceinline__ __device__ double atomic_exch(double *addr, double v) {
  return atomicExch((unsigned long long int *)addr, __double_as_longlong(v));
}
#endif

template <typename T> XINLINE T square(T v) { return v * v; }

template <typename T> XINLINE T clamp(T v, T min_v, T max_v) {
  v = v < min_v ? min_v : v;
  v = v > max_v ? max_v : v;
  return v;
}

template <typename T> XINLINE void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <typename T> XINLINE T min3(T a, T b, T c) {
  return min(min(a, b), c);
}
template <typename T> XINLINE T max3(T a, T b, T c) {
  return max(max(a, b), c);
}

template <typename scalar_t, typename FunT>
XDEVICE void for_each_pixel_near_point(scalar_t py, scalar_t px, int out_h,
                                       int out_w, scalar_t radius,
                                       FunT callback) {
  int min_x = clamp<int>(floor(px - radius), 0, out_w - 1);
  int max_x = clamp<int>(ceil(px + radius), 0, out_w - 1);
  int min_y = clamp<int>(floor(py - radius), 0, out_h - 1);
  int max_y = clamp<int>(ceil(py + radius), 0, out_h - 1);
  for (int x = min_x; x <= max_x; x++) {
    for (int y = min_y; y <= max_y; y++) {
      scalar_t dx = x - px;
      scalar_t dy = y - py;
      scalar_t r = sqrt(dx * dx + dy * dy);
      if (r <= radius) {
        callback(y, x, dy, dx, r);
      }
    }
  }
}
} // namespace haya_ext
