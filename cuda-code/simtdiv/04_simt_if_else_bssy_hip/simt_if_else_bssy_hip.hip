#include <hip/hip_runtime.h>

__device__ __noinline__ int then_path(int x, int y) {
  volatile int v = x + y;
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1664525 + 1013904223;
  }
  return v;
}

__device__ __noinline__ int else_path(int x, int y) {
  volatile int v = x - y;
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1103515245 + 12345;
  }
  return v;
}

__device__ __noinline__ int post_mix(int v) {
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1664525 + 1013904223;
  }
  return v;
}

extern "C" __global__ __attribute__((used)) void simt_if_else_bssy_kernel(
    const int *a, const int *b, int *out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  int x = a[gid];
  int y = b[gid];
  int r;

  if ((x ^ y) & 1) {
    r = then_path(x, y);
  } else {
    r = else_path(x, y);
  }

  r = post_mix(r);
  out[gid] = r;
}
