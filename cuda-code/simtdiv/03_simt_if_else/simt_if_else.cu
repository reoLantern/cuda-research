#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                               \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__global__ void simt_if_else_kernel(const int *a, const int *b, int *out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  int x = a[gid];
  int y = b[gid];
  if ((x & 1) == 0) {
    out[gid] = x + y;
  } else {
    out[gid] = x - y;
  }
}

int main() {
  int n = 1 << 16;
  int block = 256;
  int grid = (n + block - 1) / block;
  size_t bytes = static_cast<size_t>(n) * sizeof(int);

  std::vector<int> h_a(n);
  std::vector<int> h_b(n);
  for (int i = 0; i < n; ++i) {
    h_a[i] = i;
    h_b[i] = (i * 3) & 0xff;
  }

  int *d_a = nullptr;
  int *d_b = nullptr;
  int *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  simt_if_else_kernel<<<grid, block>>>(d_a, d_b, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_b.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));

  int sample = n < 8 ? n : 8;
  for (int i = 0; i < sample; ++i) {
    std::printf("%d %d\n", h_a[i], h_b[i]);
  }

  return 0;
}
