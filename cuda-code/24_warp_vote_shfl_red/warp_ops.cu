// warp_ops.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(_e));                                  \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__device__ __forceinline__ unsigned lane_id() {
  return threadIdx.x & 31;
}

enum : int {
  OFF_VOTE   = 0,    // [0..31]   （但这里只用少量位置）
  OFF_MATCH  = 64,   // [64..95]
  OFF_SHFL   = 128,  // [128..191]
  OFF_REDUX  = 256,  // [256..319]
};

// -------------------- VOTE --------------------
__device__ __noinline__ void do_vote(int* out, int seed) {
  unsigned m = __activemask();
  unsigned lane = lane_id();

  // predicate 依赖 seed，避免编译器把 ballot 直接折叠成常量
  int pred = (((int)lane + seed) & 3) == 0;

  unsigned ballot = __ballot_sync(m, pred);
  int anyv = __any_sync(m, pred);
  int allv = __all_sync(m, pred);

  if (lane == 0) {
    out[OFF_VOTE + 0] = (int)ballot;
    out[OFF_VOTE + 1] = anyv;
    out[OFF_VOTE + 2] = allv;
    out[OFF_VOTE + 3] = __popc(ballot);   // 便于 sanity check
  }
}

// -------------------- MATCH --------------------
__device__ __noinline__ void do_match(int* out, int seed) {
  unsigned m = __activemask();
  unsigned lane = lane_id();

#if __CUDA_ARCH__ >= 700
  // 让 warp 被分成 4 组（按 value 相等分组），并把每个 lane 的 group mask 写出来
  int v = (((int)lane + seed) & 3);
  unsigned g = __match_any_sync(m, v);
  out[OFF_MATCH + (int)lane] = (int)g; // 每个 lane 写自己的 group mask（更直观）

  // match_all：构造一个“几乎都相等，但 lane0 不同”的情况
  int pred_all = 0;
  int v_all = (lane == 0) ? (seed ^ 0x55) : (seed ^ 0x55) + 1; // 故意让 lane0 不同
  unsigned r_all = __match_all_sync(m, v_all, &pred_all);
  if (lane == 0) {
    out[OFF_MATCH + 32] = (int)r_all;     // 期望为 0
    out[OFF_MATCH + 33] = pred_all;      // 期望为 0
  }
#else
  // 低于 sm_70 没有 match 指令：写哨兵值，保证可编译运行
  if (lane < 32) out[OFF_MATCH + (int)lane] = -1;
  if (lane == 0) {
    out[OFF_MATCH + 32] = -1;
    out[OFF_MATCH + 33] = -1;
  }
#endif
}

// -------------------- SHUFFLE --------------------
__device__ __noinline__ void do_shuffle(int* out, int seed) {
  unsigned m = __activemask();
  unsigned lane = lane_id();

  int x = (int)lane + seed;

  // xor: 常用于蝶形通信（归约/scan/sort）
  int y_xor1 = __shfl_xor_sync(m, x, 1);

  // down: 常用于树形归约（lane i 从 i+delta 取）
  int y_down8 = __shfl_down_sync(m, x, 8);

  // idx: 从固定 srcLane 广播（例如 leader 广播）
  int y_from0 = __shfl_sync(m, x, 0);

  out[OFF_SHFL + (int)lane + 0]  = y_xor1;
  out[OFF_SHFL + (int)lane + 32] = y_down8;
  out[OFF_SHFL + (int)lane + 64] = y_from0;
}

// -------------------- REDUCE --------------------
__device__ __noinline__ void do_reduce(int* out, int seed) {
  unsigned m = __activemask();
  unsigned lane = lane_id();

  int x = ((int)lane + seed) & 15; // 小范围，便于你验证 sum/min/max 等

#if __CUDA_ARCH__ >= 800
  int s = __reduce_add_sync(m, x);
  int mn = __reduce_min_sync(m, x);
  int mx = __reduce_max_sync(m, x);

  out[OFF_REDUX + (int)lane + 0]  = s;
  out[OFF_REDUX + (int)lane + 32] = mn;
  out[OFF_REDUX + (int)lane + 64] = mx;
#else
  // 低于 sm_80 没有 __reduce_*_sync：写哨兵值，保证可编译运行
  out[OFF_REDUX + (int)lane + 0]  = -2;
  out[OFF_REDUX + (int)lane + 32] = -2;
  out[OFF_REDUX + (int)lane + 64] = -2;
#endif
}

// -------------------- KERNEL --------------------
__global__ void warp_ops_kernel(int* out, int seed) {
  // 建议 blockDim.x 固定 32：你研究 warp 指令最直观
  if (blockDim.x != 32) return;

  // ==== 你每次只保留一个操作，就注释掉其他调用 ====
//   do_vote(out, seed);
//   do_match(out, seed);
//   do_shuffle(out, seed);
  do_reduce(out, seed);
}

// -------------------- HOST MAIN --------------------
int main() {
  constexpr int N = 512;
  int* d_out = nullptr;
  int  h_out[N];

  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(int)));

  int seed = 12345; // 作为 kernel 参数，避免编译期常量折叠
  warp_ops_kernel<<<1, 32>>>(d_out, seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));

  // 打印少量位置做 sanity check（你主要看 PTX/SASS）
  std::printf("VOTE: ballot=0x%08x any=%d all=%d popc=%d\n",
              (unsigned)h_out[OFF_VOTE+0], h_out[OFF_VOTE+1],
              h_out[OFF_VOTE+2], h_out[OFF_VOTE+3]);

  std::printf("MATCH lane0 groupmask=0x%08x, match_all(r=%d,p=%d)\n",
              (unsigned)h_out[OFF_MATCH+0], h_out[OFF_MATCH+32], h_out[OFF_MATCH+33]);

  std::printf("SHFL lane0 xor1=%d down8=%d from0=%d\n",
              h_out[OFF_SHFL+0], h_out[OFF_SHFL+32], h_out[OFF_SHFL+64]);

  std::printf("REDUX lane0 sum=%d min=%d max=%d\n",
              h_out[OFF_REDUX+0], h_out[OFF_REDUX+32], h_out[OFF_REDUX+64]);

  return 0;
}
