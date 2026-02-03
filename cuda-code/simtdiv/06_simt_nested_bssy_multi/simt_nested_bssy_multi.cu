#include <cuda_runtime.h>

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

__device__ __noinline__ int outer_then(int x, int y) {
  int v = x + y;
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1664525 + 1013904223;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__device__ __noinline__ int outer_else(int x, int y) {
  int v = x - y;
#pragma unroll 1
  for (int k = 0; k < 4; ++k) {
    v = v * 1103515245 + 12345;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__device__ __noinline__ int inner_then_a(int v) {
  v ^= 0x5a5a5a5a;
#pragma unroll 1
  for (int k = 0; k < 3; ++k) {
    v = v * 134775813 + 1;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__device__ __noinline__ int inner_else_a(int v) {
  v += 0x1234;
#pragma unroll 1
  for (int k = 0; k < 3; ++k) {
    v = v * 22695477 + 1;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__device__ __noinline__ int inner_then_b(int v) {
  v ^= 0x7f4a7c15;
#pragma unroll 1
  for (int k = 0; k < 3; ++k) {
    v = v * 214013 + 2531011;
    asm volatile("" : "+r"(v));
  }
  return v;
}

__device__ __noinline__ int inner_else_b(int v) {
  v -= 0x2468;
#pragma unroll 1
  for (int k = 0; k < 3; ++k) {
    v = v * 48271 + 1;
    asm volatile("" : "+r"(v));
  }
  return v;
}

// 内层分支汇合后，仍在 outer-then 分支内继续执行
__device__ __noinline__ int after_inner_then(int v) {
#pragma unroll 1
  for (int k = 0; k < 3; ++k) {
    v = v * 1664525 + 1013904223;
    asm volatile("" : "+r"(v));
  }
  return v;
}

// 内层分支汇合后，仍在 outer-else 分支内继续执行
__device__ __noinline__ int after_inner_else(int v) {
#pragma unroll 1
  for (int k = 0; k < 3; ++k) {
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

__global__ void simt_nested_bssy_multi_kernel(const int *a, const int *b,
                                              int *out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int x = a[gid];
  int y = b[gid];
  int r;

  if ((x ^ y) & 1) {
    r = outer_then(x, y);
    if (r & 2) {
      r = inner_then_a(r);
    } else {
      r = inner_else_a(r);
    }
    // 内层分支先汇合，然后在 outer-then 内继续执行
    r = after_inner_then(r);
  } else {
    r = outer_else(x, y);
    if (r & 4) {
      r = inner_then_b(r);
    } else {
      r = inner_else_b(r);
    }
    // 内层分支先汇合，然后在 outer-else 内继续执行
    r = after_inner_else(r);
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

  simt_nested_bssy_multi_kernel<<<grid, block>>>(a, b, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(out));

  return 0;
}
