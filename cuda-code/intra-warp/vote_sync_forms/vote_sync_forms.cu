// vote_sync_forms.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers: each PTX vote.sync form in its own function ----

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.ALL P0, P0 ;                          /* 0x0000000000ff7806 */
// /* 0x000fc80000000000 */
// /*0100*/                   SEL R7, RZ, 0x1, !P0 ;                     /* 0x00000001ff077807 */
// /* 0x000fca0004000000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_all_pred(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in, p_out;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.all.pred p_out, p_in, %2;\n"
                 "  selp.u32 %0, 1, 0, p_out;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.ANY P0, P0 ;                          /* 0x0000000000ff7806 */
// /* 0x000fc80000000100 */
// /*0100*/                   SEL R7, RZ, 0x1, !P0 ;                     /* 0x00000001ff077807 */
// /* 0x000fca0004000000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_any_pred(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in, p_out;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.any.pred p_out, p_in, %2;\n"
                 "  selp.u32 %0, 1, 0, p_out;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.EQ P0, P0 ;                           /* 0x0000000000ff7806 */
// /* 0x000fc80000000200 */
// /*0100*/                   SEL R7, RZ, 0x1, !P0 ;                     /* 0x00000001ff077807 */
// /* 0x000fca0004000000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_uni_pred(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in, p_out;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.uni.pred p_out, p_in, %2;\n"
                 "  selp.u32 %0, 1, 0, p_out;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.ALL P0, !P0 ;                         /* 0x0000000000ff7806 */
// /* 0x000fc80004000000 */
// /*0100*/                   SEL R7, RZ, 0x1, !P0 ;                     /* 0x00000001ff077807 */
// /* 0x000fca0004000000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_all_pred_not(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in, p_out;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.all.pred p_out, !p_in, %2;\n"
                 "  selp.u32 %0, 1, 0, p_out;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.ANY P0, !P0 ;                         /* 0x0000000000ff7806 */
// /* 0x000fc80004000100 */
// /*0100*/                   SEL R7, RZ, 0x1, !P0 ;                     /* 0x00000001ff077807 */
// /* 0x000fca0004000000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_any_pred_not(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in, p_out;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.any.pred p_out, !p_in, %2;\n"
                 "  selp.u32 %0, 1, 0, p_out;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.EQ P0, !P0 ;                          /* 0x0000000000ff7806 */
// /* 0x000fc80004000200 */
// /*0100*/                   SEL R7, RZ, 0x1, !P0 ;                     /* 0x00000001ff077807 */
// /* 0x000fca0004000000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_uni_pred_not(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in, p_out;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.uni.pred p_out, !p_in, %2;\n"
                 "  selp.u32 %0, 1, 0, p_out;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.ANY R7, PT, P0 ;                      /* 0x0000000000077806 */
// /* 0x000fca00000e0100 */
// /*0100*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_ballot_b32(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.ballot.b32 %0, p_in, %2;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

// SASS (sm_103a, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
// /* 0x002e22000c1f5900 */
// /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00c0*/                   LOP3.LUT R0, R0, UR6, R2, 0x96, !PT ;      /* 0x0000000600007c12 */
// /* 0x001fc8000f8e9602 */
// /*00d0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000100ff7812 */
// /* 0x000fe4000780c0ff */
// /*00e0*/                   VOTE.ANY R0, PT, PT ;                      /* 0x0000000000007806 */
// /* 0x000fd600038e0100 */
// /*00f0*/                   VOTE.ANY R7, PT, !P0 ;                     /* 0x0000000000077806 */
// /* 0x000fca00040e0100 */
// /*0100*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t vote_sync_ballot_b32_not(uint32_t mask, uint32_t pred_in) {
    uint32_t out;
    asm volatile("{\n"
                 "  .reg .pred p_in;\n"
                 "  setp.ne.u32 p_in, %1, 0;\n"
                 "  vote.sync.ballot.b32 %0, !p_in, %2;\n"
                 "}\n"
                 : "=r"(out)
                 : "r"(pred_in), "r"(mask));
    return out;
}

extern "C" __global__ void vote_sync_forms_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t lane = (uint32_t)(threadIdx.x & 31);

    uint32_t a = ((const volatile uint32_t*)in)[tid];
    uint32_t mask = __activemask();

    uint32_t pred_in = (a ^ seed ^ lane) & 1u;

    uint32_t acc = 0u;
    acc ^= (vote_sync_all_pred(mask, pred_in) << 0);
    acc ^= (vote_sync_any_pred(mask, pred_in) << 1);
    acc ^= (vote_sync_uni_pred(mask, pred_in) << 2);
    acc ^= (vote_sync_all_pred_not(mask, pred_in) << 3);
    acc ^= (vote_sync_any_pred_not(mask, pred_in) << 4);
    acc ^= (vote_sync_uni_pred_not(mask, pred_in) << 5);
    acc ^= vote_sync_ballot_b32(mask, pred_in);
    acc ^= (vote_sync_ballot_b32_not(mask, pred_in) << 1);

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
        in[i] = (uint32_t)((i * 7 + 3) ^ 0x5a5a1234u);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    vote_sync_forms_kernel<<<grid, block>>>(in, out, 7u);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
