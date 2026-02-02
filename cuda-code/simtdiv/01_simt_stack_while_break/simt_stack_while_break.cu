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

__global__ void simt_stack_kernel(const int *a, int *b, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  int i = a[gid];
  int j = b[gid];
  while (i > 0) {
    if (j > 2 * i) {
      b[gid] += i;
    } else {
      break;
    }
    --i;
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
    h_a[i] = (i % 11) + 1;
    h_b[i] = (i % 17) + 1;
  }

  int *d_a = nullptr;
  int *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  simt_stack_kernel<<<grid, block>>>(d_a, d_b, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));

  int sample = n < 8 ? n : 8;
  for (int i = 0; i < sample; ++i) {
    std::printf("%d %d\n", h_a[i], h_b[i]);
  }

  return 0;
}
