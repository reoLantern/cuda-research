#include <hip/hip_runtime.h>

extern "C" __global__ __attribute__((used)) void simt_if_else_kernel(
    const int *a, const int *b, int *out, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) {
    return;
  }

  int x = a[gid];
  int y = b[gid];
  if ((x & 1) == 0) {
    out[gid] = x + y;
  } else {
    out[gid] = x - y;
  }
}
