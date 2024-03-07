#include "cuda_fp16.h"
#include <cstdint>

// CUDA使用不同size的wgmma测试

__global__ void wgmma_test1(float *gm_cd, uint64_t *desc)
{
  uint64_t a_desc = desc[0], b_desc = desc[1], c_desc = desc[2], d_desc = desc[3];
  float d_array[16];
  // uint64_t a_array[16];
  // uint64_t b_array[16];
  // for (int i = 0; i < 16; ++i)
  // {
  //   a_array[i] = desc[i * 2];
  //   b_array[i] = desc[i * 2 + 1];
  // }
  for (int i = 0; i < 16; ++i)
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

  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile("{\n\t"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n\t"
               "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9,1,1,1,0,0;\n\t"
               "wgmma.commit_group.sync.aligned;"
               "wgmma.wait_group.sync.aligned 1;"
               "}\n\t"
               : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]), "+f"(d_array[3]) "+f"(d_array[4]), "+f"(d_array[5]), "+f"(d_array[6]), "+f"(d_array[7])
               : "l"(a_desc), "l"(b_desc)
               :);

  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile("{\n\t"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n\t"
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17,1,1,1,0,0;\n\t"
               "wgmma.commit_group.sync.aligned;"
               "wgmma.wait_group.sync.aligned 1;"
               "}\n\t"
               : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]), "+f"(d_array[3]) "+f"(d_array[4]), "+f"(d_array[5]), "+f"(d_array[6]), "+f"(d_array[7]),
                 "+f"(d_array[8]), "+f"(d_array[9]), "+f"(d_array[10]), "+f"(d_array[11]), "+f"(d_array[12]), "+f"(d_array[13]), "+f"(d_array[14]), "+f"(d_array[15])
               : "l"(a_desc), "l"(b_desc)
               :);

  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile("{\n\t"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n\t"
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33,1,1,1,0,0;\n\t"
               "wgmma.commit_group.sync.aligned;"
               "wgmma.wait_group.sync.aligned 1;"
               "}\n\t"
               : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]), "+f"(d_array[3]) "+f"(d_array[4]), "+f"(d_array[5]), "+f"(d_array[6]), "+f"(d_array[7]),
                 "+f"(d_array[8]), "+f"(d_array[9]), "+f"(d_array[10]), "+f"(d_array[11]), "+f"(d_array[12]), "+f"(d_array[13]), "+f"(d_array[14]), "+f"(d_array[15]),
                 "+f"(d_array[16]), "+f"(d_array[17]), "+f"(d_array[18]), "+f"(d_array[19]), "+f"(d_array[20]), "+f"(d_array[21]), "+f"(d_array[22]), "+f"(d_array[23]),
                 "+f"(d_array[24]), "+f"(d_array[25]), "+f"(d_array[26]), "+f"(d_array[27]), "+f"(d_array[28]), "+f"(d_array[29]), "+f"(d_array[30]), "+f"(d_array[31])
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