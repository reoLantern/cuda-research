// intra_warp.cu
//
// Warp-level intrinsics for SASS inspection:
//   vote:   __all_sync / __any_sync / __ballot_sync
//   match:  __match_any_sync / __match_all_sync
//   reduce: __reduce_{add,min,max}_sync
//   shuffle: __shfl_sync / __shfl_{up,down,xor}_sync
//
// Each kernel isolates one intrinsic; select with argv[1].

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#ifndef ONLY_CASE
#define ONLY_CASE -1
#endif

#define CASE_ENABLED(N) ((ONLY_CASE) < 0 || (ONLY_CASE) == (N))

__device__ __forceinline__ unsigned lane_id() {
    return threadIdx.x & 31;
}

// ---- vote ----
#if CASE_ENABLED(0)
// PTX (warp_all_sync_kernel):
//   and.b32 %r8, %r7, 1;
//   setp.ne.b32 %p1, %r8, 0;
//   vote.sync.all.pred %p2, %p1, %r1;
//   selp.b32 %r10, 1, 0, %p2;
// SASS (sm_103a, isolated kernel):
//   /*0090*/ LOP3.LUT P0, RZ, R4, 0x1, RZ, 0xc0, !PT ;
//   /*00a0*/ VOTE.ANY R4, PT, PT ;
//   /*00b0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
//   /*00c0*/ VOTE.ALL P0, P0 ;
//   /*00d0*/ SEL R5, RZ, 0x1, !P0 ;
__device__ __forceinline__ int warp_all_sync(unsigned mask, int pred) {
    return __all_sync(mask, pred);
}
#endif

#if CASE_ENABLED(1)
// PTX (warp_any_sync_kernel):
//   and.b32 %r8, %r7, 1;
//   setp.ne.b32 %p1, %r8, 0;
//   vote.sync.any.pred %p2, %p1, %r1;
//   selp.b32 %r10, 1, 0, %p2;
// SASS (sm_103a, isolated kernel):
//   /*0090*/ LOP3.LUT P0, RZ, R4, 0x1, RZ, 0xc0, !PT ;
//   /*00a0*/ VOTE.ANY R4, PT, PT ;
//   /*00b0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
//   /*00c0*/ VOTE.ANY P0, P0 ;
//   /*00d0*/ SEL R5, RZ, 0x1, !P0 ;
__device__ __forceinline__ int warp_any_sync(unsigned mask, int pred) {
    return __any_sync(mask, pred);
}
#endif

#if CASE_ENABLED(2)
// PTX (warp_ballot_sync_kernel):
//   and.b32 %r8, %r7, 3;
//   setp.eq.s32 %p1, %r8, 0;
//   vote.sync.ballot.b32 %r9, %p1, %r1;
// SASS (sm_103a, isolated kernel):
//   /*0090*/ VOTE.ANY R0, PT, PT ;
//   /*00a0*/ LOP3.LUT P0, RZ, R4, 0x3, RZ, 0xc0, !PT ;
//   /*00b0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
//   /*00c0*/ VOTE.ANY R5, PT, !P0 ;
__device__ __forceinline__ unsigned warp_ballot_sync(unsigned mask, int pred) {
    return __ballot_sync(mask, pred);
}
#endif

// ---- match ----
#if CASE_ENABLED(3)
// PTX (warp_match_any_u32_kernel):
//   and.b32 %r8, %r7, 3;
//   match.any.sync.b32 %r9, %r8, %r1;
// SASS (sm_103a, isolated kernel):
//   /*00a0*/ LOP3.LUT R7, R4, 0x3, RZ, 0xc0, !PT ;
//   /*00b0*/ MATCH.ANY R7, R7 ;
//   /*00c0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
__device__ __forceinline__ unsigned warp_match_any_sync_u32(unsigned mask, unsigned value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return __match_any_sync(mask, value);
#else
    (void)mask;
    (void)value;
    return 0u;
#endif
}
#endif

#if CASE_ENABLED(4)
// PTX (warp_match_any_u64_kernel):
//   and.b32 %r8, %r7, 3;
//   cvt.u64.u32 %rd3, %r8;
//   match.any.sync.b64 %r9, %rd3, %r1;
// SASS (sm_103a, isolated kernel):
//   /*00b0*/ LOP3.LUT R2, R2, 0x3, RZ, 0xc0, !PT ;
//   /*00c0*/ MATCH.ANY.U64 R9, R2 ;
//   /*00d0*/ IMAD.WIDE R4, R7, 0x4, R4 ;
__device__ __forceinline__ unsigned warp_match_any_sync_u64(unsigned mask, unsigned long long value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return __match_any_sync(mask, value);
#else
    (void)mask;
    (void)value;
    return 0u;
#endif
}
#endif

#if CASE_ENABLED(5)
// PTX (warp_match_all_u32_kernel):
//   xor.b32 %r9, %r2, %r8;
//   match.all.sync.b32 %r10|%p2, %r9, %r1;
//   selp.b32 %r11, 1, 0, %p2;
// SASS (sm_103a, isolated kernel):
//   /*00c0*/ VOTE.ANY R0, PT, PT ;
//   /*00d0*/ MATCH.ALL P0, R9, R9 ;
//   /*00e0*/ IMAD.WIDE R2, R7, 0x4, R2 ;
//   /*0100*/ SEL R7, RZ, 0x1, !P0 ;
__device__ __forceinline__ unsigned warp_match_all_sync_u32(unsigned mask, unsigned value, int* pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return __match_all_sync(mask, value, pred);
#else
    (void)mask;
    (void)value;
    if (pred) {
        *pred = 0;
    }
    return 0u;
#endif
}
#endif

// ---- reduce ----
#if CASE_ENABLED(6)
// PTX (warp_reduce_add_kernel):
//   and.b32 %r8, %r7, 15;
//   redux.sync.add.s32 %r9, %r8, %r1;
// SASS (sm_103a, isolated kernel):
//   /*0090*/ LOP3.LUT R7, R2, 0xf, RZ, 0xc0, !PT ;
//   /*00b0*/ REDUX.SUM.S32 UR7, R7 ;
//   /*00d0*/ MOV R5, UR7 ;
__device__ __forceinline__ int warp_reduce_add_sync(unsigned mask, int value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_add_sync(mask, value);
#else
    (void)mask;
    return value;
#endif
}
#endif

#if CASE_ENABLED(7)
// PTX (warp_reduce_min_kernel):
//   and.b32 %r8, %r7, 15;
//   redux.sync.min.s32 %r9, %r8, %r1;
// SASS (sm_103a, isolated kernel):
//   /*0090*/ LOP3.LUT R5, R4, 0xf, RZ, 0xc0, !PT ;
//   /*00a0*/ CREDUX.MIN.S32 UR7, R5 ;
//   /*00c0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
__device__ __forceinline__ int warp_reduce_min_sync(unsigned mask, int value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_min_sync(mask, value);
#else
    (void)mask;
    return value;
#endif
}
#endif

#if CASE_ENABLED(8)
// PTX (warp_reduce_max_kernel):
//   and.b32 %r8, %r7, 15;
//   redux.sync.max.s32 %r9, %r8, %r1;
// SASS (sm_103a, isolated kernel):
//   /*0090*/ LOP3.LUT R5, R4, 0xf, RZ, 0xc0, !PT ;
//   /*00a0*/ CREDUX.MAX.S32 UR7, R5 ;
//   /*00c0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
__device__ __forceinline__ int warp_reduce_max_sync(unsigned mask, int value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_max_sync(mask, value);
#else
    (void)mask;
    return value;
#endif
}
#endif

// ---- shuffle ----
#if CASE_ENABLED(9)
// PTX (warp_shfl_sync_kernel):
//   and.b32 %r9, %r2, 31;
//   shfl.sync.idx.b32 %r10|%p1, %r8, %r9, 31, %r1;
// SASS (sm_103a, isolated kernel):
//   /*00a0*/ LOP3.LUT R4, R9, 0x1f, RZ, 0xc0, !PT ;
//   /*00c0*/ SHFL.IDX PT, R7, R7, R4, 0x1f ;
//   /*00d0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
__device__ __forceinline__ int warp_shfl_sync(unsigned mask, int value, int src_lane) {
    return __shfl_sync(mask, value, src_lane, 32);
}
#endif

#if CASE_ENABLED(10)
// PTX (warp_shfl_up_kernel):
//   and.b32 %r9, %r2, 7;
//   add.s32 %r10, %r9, 1;
//   shfl.sync.up.b32 %r11|%p1, %r8, %r10, 0, %r1;
// SASS (sm_103a, isolated kernel):
//   /*00b0*/ VOTE.ANY R0, PT, PT ;
//   /*00d0*/ SHFL.UP PT, R7, R7, R4, RZ ;
//   /*00e0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
__device__ __forceinline__ int warp_shfl_up_sync(unsigned mask, int value, int delta) {
    return __shfl_up_sync(mask, value, (unsigned)delta, 32);
}
#endif

#if CASE_ENABLED(11)
// PTX (warp_shfl_down_kernel):
//   and.b32 %r9, %r2, 7;
//   add.s32 %r10, %r9, 1;
//   shfl.sync.down.b32 %r11|%p1, %r8, %r10, 31, %r1;
// SASS (sm_103a, isolated kernel):
//   /*00b0*/ VOTE.ANY R0, PT, PT ;
//   /*00d0*/ SHFL.DOWN PT, R7, R7, R4, 0x1f ;
//   /*00e0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
__device__ __forceinline__ int warp_shfl_down_sync(unsigned mask, int value, int delta) {
    return __shfl_down_sync(mask, value, (unsigned)delta, 32);
}
#endif

#if CASE_ENABLED(12)
// PTX (warp_shfl_xor_kernel):
//   and.b32 %r9, %r2, 3;
//   shl.b32 %r11, %r10, %r9;
//   shfl.sync.bfly.b32 %r12|%p1, %r8, %r11, 31, %r1;
// SASS (sm_103a, isolated kernel):
//   /*00c0*/ SHF.L.U32 R4, R8, UR4, RZ ;
//   /*00e0*/ SHFL.BFLY PT, R7, R7, R4, 0x1f ;
//   /*00f0*/ IMAD.WIDE R2, R5, 0x4, R2 ;
__device__ __forceinline__ int warp_shfl_xor_sync(unsigned mask, int value, int lane_mask) {
    return __shfl_xor_sync(mask, value, lane_mask, 32);
}
#endif

// ---- kernels ----
#if CASE_ENABLED(0)
extern "C" __global__ void warp_all_sync_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int pred = ((lane + seed) & 1);
    out[tid] = warp_all_sync(mask, pred);
}
#endif

#if CASE_ENABLED(1)
extern "C" __global__ void warp_any_sync_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int pred = ((lane + seed) & 1);
    out[tid] = warp_any_sync(mask, pred);
}
#endif

#if CASE_ENABLED(2)
extern "C" __global__ void warp_ballot_sync_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int pred = (((lane + seed) & 3) == 0);
    out[tid] = (int)warp_ballot_sync(mask, pred);
}
#endif

#if CASE_ENABLED(3)
extern "C" __global__ void warp_match_any_u32_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    unsigned value = (unsigned)((lane + seed) & 3);
    out[tid] = (int)warp_match_any_sync_u32(mask, value);
}
#endif

#if CASE_ENABLED(4)
extern "C" __global__ void warp_match_any_u64_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    unsigned long long value = (unsigned long long)((lane + seed) & 3);
    out[tid] = (int)warp_match_any_sync_u64(mask, value);
}
#endif

#if CASE_ENABLED(5)
extern "C" __global__ void warp_match_all_u32_kernel(int* out_mask, int* out_pred, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    unsigned value = (unsigned)(seed & 0xff);
    if (lane == 0) {
        value ^= 1u;
    }
    int pred = 0;
    unsigned m = warp_match_all_sync_u32(mask, value, &pred);
    out_mask[tid] = (int)m;
    out_pred[tid] = pred;
}
#endif

#if CASE_ENABLED(6)
extern "C" __global__ void warp_reduce_add_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int value = ((lane + seed) & 15);
    out[tid] = warp_reduce_add_sync(mask, value);
}
#endif

#if CASE_ENABLED(7)
extern "C" __global__ void warp_reduce_min_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int value = ((lane + seed) & 15);
    out[tid] = warp_reduce_min_sync(mask, value);
}
#endif

#if CASE_ENABLED(8)
extern "C" __global__ void warp_reduce_max_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int value = ((lane + seed) & 15);
    out[tid] = warp_reduce_max_sync(mask, value);
}
#endif

#if CASE_ENABLED(9)
extern "C" __global__ void warp_shfl_sync_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int value = lane + seed;
    int src_lane = seed & 31;
    out[tid] = warp_shfl_sync(mask, value, src_lane);
}
#endif

#if CASE_ENABLED(10)
extern "C" __global__ void warp_shfl_up_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int value = lane + seed;
    int delta = (seed & 7) + 1;
    out[tid] = warp_shfl_up_sync(mask, value, delta);
}
#endif

#if CASE_ENABLED(11)
extern "C" __global__ void warp_shfl_down_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int value = lane + seed;
    int delta = (seed & 7) + 1;
    out[tid] = warp_shfl_down_sync(mask, value, delta);
}
#endif

#if CASE_ENABLED(12)
extern "C" __global__ void warp_shfl_xor_kernel(int* out, int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned mask = __activemask();
    int lane = (int)lane_id();
    int value = lane + seed;
    int lane_mask = 1 << (seed & 3);
    out[tid] = warp_shfl_xor_sync(mask, value, lane_mask);
}
#endif

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 32;
    int which = 0;
    int seed = 12345;

#if ONLY_CASE >= 0
    which = ONLY_CASE;
#endif

    if (argc > 1) {
        which = std::atoi(argv[1]);
    }
    if (argc > 2) {
        seed = std::atoi(argv[2]);
    }

    int* out0 = nullptr;
    int* out1 = nullptr;
    ck(cudaMallocManaged(&out0, N * sizeof(int)), "cudaMallocManaged out0");
    ck(cudaMallocManaged(&out1, N * sizeof(int)), "cudaMallocManaged out1");
    ck(cudaMemset(out0, 0, N * sizeof(int)), "cudaMemset out0");
    ck(cudaMemset(out1, 0, N * sizeof(int)), "cudaMemset out1");

    dim3 block(32);
    dim3 grid(1);

    switch (which) {
#if CASE_ENABLED(0)
        case 0:
            warp_all_sync_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(1)
        case 1:
            warp_any_sync_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(2)
        case 2:
            warp_ballot_sync_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(3)
        case 3:
            warp_match_any_u32_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(4)
        case 4:
            warp_match_any_u64_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(5)
        case 5:
            warp_match_all_u32_kernel<<<grid, block>>>(out0, out1, seed);
            break;
#endif
#if CASE_ENABLED(6)
        case 6:
            warp_reduce_add_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(7)
        case 7:
            warp_reduce_min_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(8)
        case 8:
            warp_reduce_max_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(9)
        case 9:
            warp_shfl_sync_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(10)
        case 10:
            warp_shfl_up_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(11)
        case 11:
            warp_shfl_down_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
#if CASE_ENABLED(12)
        case 12:
            warp_shfl_xor_kernel<<<grid, block>>>(out0, seed);
            break;
#endif
        default:
            std::fprintf(stderr, "case out of range (0-12)\n");
            cudaFree(out0);
            cudaFree(out1);
            return 1;
    }

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("case %d lane0 out0=0x%08x out1=%d\n",
                which, (unsigned)out0[0], out1[0]);

    cudaFree(out0);
    cudaFree(out1);
    return 0;
}
