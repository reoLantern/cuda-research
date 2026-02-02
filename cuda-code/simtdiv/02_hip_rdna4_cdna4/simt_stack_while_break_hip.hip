#include <hip/hip_runtime.h>

extern "C" __global__ __attribute__((used)) void simt_stack_kernel(const int *a,
                                                                    int *b,
                                                                    int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) {
    return;
  }

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
