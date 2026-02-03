#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                               \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__device__ __noinline__ int then_path(int x, int y) {
  int v = x + y;
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1664525 + 1013904223;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__device__ __noinline__ int else_path(int x, int y) {
  int v = x - y;
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1103515245 + 12345;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__device__ __noinline__ int post_mix(int v) {
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1664525 + 1013904223;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__global__ void simt_if_else_bssy_kernel(const int *a, const int *b, int *out) {
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

int main() {
  int n = 1 << 16;
  int block = 256;
  int grid = n / block;
  size_t bytes = static_cast<size_t>(n) * sizeof(int);

  int *a = nullptr;
  int *b = nullptr;
  int *out = nullptr;
  CUDA_CHECK(cudaMallocManaged(&a, bytes));
  CUDA_CHECK(cudaMallocManaged(&b, bytes));
  CUDA_CHECK(cudaMallocManaged(&out, bytes));

  for (int i = 0; i < n; ++i) {
    a[i] = i * 3 + 1;
    b[i] = (i * 7) ^ 0x5a5a5a5a;
  }

  simt_if_else_bssy_kernel<<<grid, block>>>(a, b, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(out));

  return 0;
}
