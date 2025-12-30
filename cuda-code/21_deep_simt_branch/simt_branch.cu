#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                    \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

__device__ __noinline__ int pathA(volatile int* scratch, int sid, int x, int lane) {
  // 做多一点工作，避免被完全谓词化/消掉
  #pragma unroll 1
  for (int i = 0; i < 6; ++i) {
    x = (x ^ (lane + i)) + (x << 1);
    if (((lane + i) & 1) && (x & 3)) x += 7;
    scratch[sid] = x; // volatile side-effect
  }
  return x;
}

__device__ __noinline__ int pathB(volatile int* scratch, int sid, int x, int lane) {
  #pragma unroll 1
  for (int i = 0; i < 6; ++i) {
    x = (x + (lane * 3 + i)) ^ (x >> 1);
    if (((lane + i) & 2) && (x & 1)) x ^= 0x5a5a;
    scratch[sid] = x;
  }
  return x;
}

__device__ __noinline__ int pathC(volatile int* scratch, int sid, int x, int lane) {
  // 带 break/continue 的循环控制流，便于观察 reconvergence
  #pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    if (((lane + i) & 1) == 1) {
      x += i;
      scratch[sid] = x;
      continue;
    }
    if (((lane + i) & 2) == 2) {
      x ^= (i * 17);
      scratch[sid] = x;
      break;
    }
    x = x + (x ^ (lane + 11));
    scratch[sid] = x;
  }
  return x;
}

__device__ __noinline__ int pathD(volatile int* scratch, int sid, int x, int lane) {
  // 小 switch 也会形成结构化 CFG（不同架构生成细节不同）
  int sel = (lane ^ x) & 3;
  switch (sel) {
    case 0: x = x + 1; break;
    case 1: x = x ^ 0x1234; break;
    case 2: x = x - 7; break;
    default: x = x + (lane * 5); break;
  }
  scratch[sid] = x;
  return x;
}

__global__ void nested_simt(const int* __restrict__ in,
                           int* __restrict__ out,
                           volatile int* __restrict__ scratch,
                           int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int lane = threadIdx.x & 31;
  int x = in[tid];

  // 给每线程留一段 scratch，便于 side-effect，避免过度优化
  int base = tid * 8;

  // ===== Level 1: warp 分成两半（强分歧）=====
  if (lane < 16) {
    x = pathA(scratch, base + 0, x, lane);

    // ===== Level 2: 再按 bit 分裂（嵌套分歧）=====
    if (lane & 4) {
      x = pathB(scratch, base + 1, x, lane);

      // ===== Level 3: 进一步分歧 + 循环控制流 =====
      if (((lane ^ x) & 1) == 0) {
        x = pathC(scratch, base + 2, x, lane);
      } else {
        x = pathD(scratch, base + 3, x, lane);
      }
    } else {
      x = pathC(scratch, base + 4, x, lane);

      if (((lane + x) & 2) == 2) {
        x = pathD(scratch, base + 5, x, lane);
      } else {
        x = pathB(scratch, base + 6, x, lane);
      }
    }
  } else {
    x = pathB(scratch, base + 7, x, lane);

    // ===== Level 2 (else half): 不同结构的嵌套 =====
    if (lane & 8) {
      x = pathD(scratch, base + 0, x, lane);
      x = pathC(scratch, base + 1, x, lane);
    } else {
      x = pathC(scratch, base + 2, x, lane);
      x = pathA(scratch, base + 3, x, lane);
    }
  }

  out[tid] = x;
}

int main() {
  constexpr int block = 128;
  constexpr int n = 4096;
  constexpr int grid = (n + block - 1) / block;

  int *d_in = nullptr, *d_out = nullptr;
  volatile int* d_scratch = nullptr;

  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  // 每线程 8 个 int scratch
  CUDA_CHECK(cudaMalloc((void**)&d_scratch, (size_t)n * 8 * sizeof(int)));

  // 初始化输入
  int* h_in = (int*)malloc(n * sizeof(int));
  for (int i = 0; i < n; ++i) h_in[i] = i * 13 + 7;
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice));

  nested_simt<<<grid, block>>>(d_in, d_out, d_scratch, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 拉回少量结果，防止“未使用”之类的极端情况
  int h_out[8];
  CUDA_CHECK(cudaMemcpy(h_out, d_out, 8 * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 8; ++i) printf("out[%d]=%d\n", i, h_out[i]);

  free(h_in);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree((void*)d_scratch));
  return 0;
}
