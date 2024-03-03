#include "cuda_fp16.h"
#include <cstdint>

__global__ void wgmma_test1(float *gm_cd, uint64_t *desc)
{
    uint64_t a_desc = desc[0], b_desc = desc[1], c_desc = desc[2], d_desc = desc[3];
    float d_array[8];
    for (int i = 0; i < 8; ++i)
    {
        d_array[i] = gm_cd[i];
    }
    asm volatile("{\n\t"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n\t"
                 "{%0, %1, %2, %3}, %8, %9,1,1,1,0,0;\n\t"
                 "wgmma.commit_group.sync.aligned;"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n\t"
                 "{%4, %5, %6, %7}, %10, %11,1,1,1,0,0;\n\t"
                 "wgmma.commit_group.sync.aligned;"
                 "}\n\t"
                 : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]), "+f"(d_array[3]),
                   "+f"(d_array[4]), "+f"(d_array[5]), "+f"(d_array[6]), "+f"(d_array[7])
                 : "l"(a_desc), "l"(b_desc), "l"(c_desc), "l"(d_desc)
                 :);

    a_desc++;
    b_desc++;
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile("{\n\t"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n\t"
                 "{%0, %1, %2, %3}, %4, %5,1,1,1,0,0;\n\t"
                 "wgmma.commit_group.sync.aligned;"
                 "wgmma.wait_group.sync.aligned 1;"
                 "}\n\t"
                 : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]),
                   "+f"(d_array[3])
                 : "l"(a_desc), "l"(b_desc)
                 :);

    for (int i = 0; i < 8; ++i)
    {
        gm_cd[i] = d_array[i];
    }
}

int main()
{
    return 0;
}