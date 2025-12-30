#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__device__ __forceinline__ uint32_t xs32(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

__device__ __noinline__ uint32_t heavy(uint32_t r, int lane, int tag,
                                       volatile int* scratch, int idx) {
  #pragma unroll 1
  for (int i = 0; i < 6; ++i) {
    r = xs32(r + (uint32_t)(0x9e3779b9u * (tag + 1) + lane + i));
    if ((r ^ (uint32_t)lane) & 1u) r += 0x7f4a7c15u;
    scratch[idx] = (int)r; // volatile side-effect
  }
  return r;
}

// 互递归：避免编译器把它轻易变形为简单循环
__device__ __noinline__ uint32_t f(uint32_t r, int depth, int lane,
                                  volatile int* scratch, int base);
__device__ __noinline__ uint32_t g(uint32_t r, int depth, int lane,
                                  volatile int* scratch, int base);

__device__ __noinline__ uint32_t f(uint32_t r, int depth, int lane,
                                  volatile int* scratch, int base) {
  if (depth <= 0) return r;

  // 关键：warp 内分歧条件（动态、lane相关）
  bool c = (((r >> (depth & 31)) ^ (uint32_t)lane) & 1u) != 0u;

  if (c) {
    r = heavy(r, lane, 10 + depth, scratch, base + (depth & 63));
    // 关键：在分歧路径里 CALL，且 join 在 CALL 之后发生
    if (((r ^ (uint32_t)lane) & 3u) != 0u) {
      r = g(r, depth - 1, lane, scratch, base);
    }
  } else {
    r = heavy(r, lane, 110 + depth, scratch, base + (depth & 63));
    if (((r + (uint32_t)lane) & 7u) == 0u) {
      r = g(r, depth - 1, lane, scratch, base);
    }
  }

  // join 后还有共同代码：避免被整体合并成“一大块”
  r ^= (0x13579bdfu + (uint32_t)depth);
  scratch[base + 64 + (depth & 63)] = (int)r;
  return r;
}

__device__ __noinline__ uint32_t g(uint32_t r, int depth, int lane,
                                  volatile int* scratch, int base) {
  if (depth <= 0) return r;

  bool c = (((r >> ((depth * 5) & 31)) + (uint32_t)lane) & 1u) != 0u;

  if (c) {
    r = heavy(r, lane, 210 + depth, scratch, base + 2 + (depth & 63));
    if ((r & 1u) && ((lane & 3) != 0)) {
      r = f(r, depth - 1, lane, scratch, base);
    }
  } else {
    r = heavy(r, lane, 310 + depth, scratch, base + 2 + (depth & 63));
    if ((r & 2u) && ((lane & 1) == 0)) {
      r = f(r, depth - 1, lane, scratch, base);
    }
  }

  r += (0x2468ace0u + (uint32_t)depth);
  scratch[base + 96 + (depth & 31)] = (int)r;
  return r;
}

__global__ void k(const uint32_t* __restrict__ in,
                  uint32_t* __restrict__ out,
                  volatile int* __restrict__ scratch,
                  int n, int max_depth_mask) {
  int tid  = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int lane = threadIdx.x & 31;
  int base = tid * 160;

  uint64_t t = clock64();
  uint32_t r = in[tid] ^ (uint32_t)t ^ (uint32_t)(t >> 32);
  r ^= (uint32_t)(tid * 0x85ebca6bu) ^ (uint32_t)(lane * 0xc2b2ae35u);

  // 每个 thread 的“递归深度”由输入决定 => 运行期不可预测（warp 内也不一致）
  int depth = (int)((r ^ (uint32_t)lane) & (uint32_t)max_depth_mask);

  // 先做一次随机扰动
  r = heavy(r, lane, 999, scratch, base);

  // 递归进入：深度完全由数据决定
  r = f(r, depth, lane, scratch, base);

  scratch[base + 159] = (int)r;
  out[tid] = r;
}

static uint32_t host_xs32(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

int main() {
  constexpr int n = 8192;
  constexpr int block = 256;
  int grid = (n + block - 1) / block;

  // 递归会用到 device stack：把栈调大一点
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 1 << 20)); // 1MB/thread（你可再调）

  uint32_t *d_in=nullptr, *d_out=nullptr;
  volatile int* d_scratch=nullptr;

  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc((void**)&d_scratch, (size_t)n * 160 * sizeof(int)));

  uint32_t* h_in = (uint32_t*)malloc(n * sizeof(uint32_t));
  uint32_t seed = 0x12345678u;
  for (int i=0;i<n;++i) { seed = host_xs32(seed + (uint32_t)i); h_in[i]=seed; }
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(uint32_t), cudaMemcpyHostToDevice));

  // depth ∈ [0, 31]（mask=31）；你也可以改成 63/127，但要注意 stack
  k<<<grid, block>>>(d_in, d_out, d_scratch, n, 31);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_out[8];
  CUDA_CHECK(cudaMemcpy(h_out, d_out, 8*sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (int i=0;i<8;++i) printf("out[%d]=0x%08x\n", i, h_out[i]);

  free(h_in);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree((void*)d_scratch));
  return 0;
}
