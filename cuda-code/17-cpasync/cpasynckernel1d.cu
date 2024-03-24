#include <cuda/barrier>
#define __cccl_lib_experimental_ctk12_cp_async_exposure

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__

#ifndef __cccl_lib_experimental_ctk12_cp_async_exposure
static_assert(false, "libcu++ does not have experimental CTK 12 cp_async feature exposure.");
#endif //  __cccl_lib_has_experimental_ctk12_cp_async_exposure

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int *data, size_t offset)
{
  // Shared memory buffers for x and y. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

// 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
//    b) Make initialized barrier visible in async proxy.

  __shared__ barrier bar;


  // 2. Initiate TMA transfer to copy global to shared memory.

  if (threadIdx.x == 0)
  {
    cde::cp_async_bulk_global_to_shared(smem_data, data + offset, sizeof(smem_data), bar);
    // 3a. Arrive on the barrier and tell how many bytes are expected to come in (the transaction count)

  }
  else
  {

  }



}
int main()
{
  return 0;
}