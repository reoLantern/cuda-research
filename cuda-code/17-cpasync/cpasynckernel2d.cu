#include <cuda.h>         // CUtensormap
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#define SMEM_WIDTH 64
#define SMEM_HEIGHT 64

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.

  __shared__ barrier bar;




  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.

  } else {
    // Other threads just arrive.

  }

}

int main()
{
  
  return 0;
}