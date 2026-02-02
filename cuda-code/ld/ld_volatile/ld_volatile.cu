// ld_volatile.cu
//
// Covers:
// ld.volatile{.ss}{.level::prefetch_size}{.vec}.type d, [a];

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00a0*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_volatile_global_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*00c0*/                   LDG.E.64.STRONG.SYS R4, desc[UR4][R4.64] ;    /* 0x0000000404047981 */
        // /* 0x001ea2000c1f5b00 */
        // /*00d0*/                   IADD3 R2, P0, PT, R2, UR10, RZ ;              /* 0x0000000a02027c10 */
        // /* 0x000fc8000ff1e0ff */
        // /*00e0*/                   IADD3.X R3, PT, PT, R3, UR11, RZ, P0, !PT ;   /* 0x0000000b03037c10 */
        // /* 0x000fe400087fe4ff */
        // /*00f0*/                   LOP3.LUT R7, R4, R5, RZ, 0x3c, !PT ;          /* 0x0000000504077212 */
        // /* 0x004fca00078e3cff */
        // /*0100*/                   STG.E desc[UR4][R2.64], R7 ;                  /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t ld_volatile_global_v2_u32(const uint32_t* p) {
    uint32_t d0, d1;
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(d0), "=r"(d1) : "l"(p));
    return d0 ^ d1;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR6][R2.64] ;   /* 0x0000000602027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   UMOV UR4, 0x400 ;                       /* 0x0000040000047882 */
        // /* 0x000fe20000000000 */
        // /*00c0*/                   SHF.L.U32 R0, R7, 0x2, RZ ;             /* 0x0000000207007819 */
        // /* 0x000fe200000006ff */
        // /*00d0*/                   ULEA UR4, UR5, UR4, 0x18 ;              /* 0x0000000405047291 */
        // /* 0x008fc6000f8ec0ff */
        // /*00e0*/                   LOP3.LUT R9, R0, 0xfc, RZ, 0xc0, !PT ;  /* 0x000000fc00097812 */
        // /* 0x000fe200078ec0ff */
        // /*00f0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;             /* 0x0000000407047825 */
        // /* 0x001fca00078e0204 */
        // /*0100*/                   STS [R9+UR4], R2 ;                      /* 0x0000000209007988 */
        // /* 0x004fe20008000804 */
        // /*0110*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;           /* 0x0000000000007b1d */
        // /* 0x000fec0000010000 */
        // /*0120*/                   LDS R11, [R9+UR4] ;                     /* 0x00000004090b7984 */
        // /* 0x000e280008000800 */
        // /*0130*/                   STG.E desc[UR6][R4.64], R11 ;           /* 0x0000000b04007986 */
        // /* 0x001fe2000c101906 */
__device__ __forceinline__ uint32_t ld_volatile_shared_u32(const uint32_t* p) {
    uint32_t addr = __cvta_generic_to_shared(p);
    uint32_t d;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(d) : "r"(addr));
    return d;
}

extern "C" __global__ void ld_volatile_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out) {
    __shared__ uint32_t shmem[64];

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = tid & 63;

    shmem[lane] = in[tid];
    __syncthreads();

    const uint32_t* p = in + tid;
    const uint32_t* p2 = in + (tid & ~1);

    uint32_t acc = 0;
    acc ^= ld_volatile_global_u32(p);
    acc ^= ld_volatile_global_v2_u32(p2);
    acc ^= ld_volatile_shared_u32(&shmem[lane]);

    out[tid] = acc;
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    uint32_t* in;
    uint32_t* out;

    ck(cudaMallocManaged(&in, N * sizeof(uint32_t)), "cudaMallocManaged in");
    ck(cudaMallocManaged(&out, N * sizeof(uint32_t)), "cudaMallocManaged out");

    for (int i = 0; i < N; ++i) {
        in[i] = (uint32_t)((i * 19 + 9) ^ 0x6b6b5a5au);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    ld_volatile_kernel<<<grid, block>>>(in, out);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
