examples/57_hopper_grouped_gemm 展示了使用TMA的例子。

```cpp
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum; // Kernel to launch
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;
```

进入CollectiveMainloop的定义，即CollectiveOp，看到它定义在`// GMMA_TMA_WS_FP8_FAST_ACCUM_SS`注释的选项下，其中WS表示Warp Specialized，SS表示A和B都来自Shared Memory；有别的选项是RS，表示A来自Register，B来自Shared Memory。

进入`cutlass::gemm::kernel::GemmUniversal`的定义后，可以看到有一个`operator()`。


上面进入了文件`include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp`，这个文件应该会比较复杂，所以我们先看一个简单的：`include/cutlass/gemm/kernel/sm90_gemm_tma.hpp`中，观察`operator()`函数，有如下代码：

```cpp
// ....
    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }
////////////////////////////////////////////////////////////////////////////////////////////////////
// 进入其他文件
////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

CUTE_HOST_DEVICE
void
prefetch_tma_descriptor(TmaDescriptor const* desc_ptr)
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state space: const or param)
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// 回到原文件
////////////////////////////////////////////////////////////////////////////////////////////////////
```

这里调用了prefetch.tensormap，根据[ptx文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu)描述:

```ptx
prefetch{.space}.level                    [a];   // prefetch to data cache
prefetch.global.level::eviction_priority  [a];   // prefetch to data cache

prefetchu.L1  [a];             // prefetch to uniform cache

prefetch{.tensormap_space}.tensormap [a];  // prefetch the tensormap

.space =                    { .global, .local };
.level =                    { .L1, .L2 };
.level::eviction_priority = { .L2::evict_last, .L2::evict_normal };
.tensormap_space =          { .const, .param };
```

> The prefetch instruction brings the cache line containing the specified address in global or local memory state space into the specified cache level.  
If the .tensormap qualifier is specified then the prefetch instruction brings the cache line containing the specified address in the .const or .param memory state space for subsequent use by the cp.async.bulk.tensor instruction.

相当于把指定地址的tensormap加载到相应的cache level。继续看：

```cpp
    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = params.mainloop.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = params.mainloop.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

////////////////////////////////////////////////////////////////////////////////////////////////////
// 进入其他文件
////////////////////////////////////////////////////////////////////////////////////////////////////

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }
```

```cpp
    // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
    auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
    auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
    auto l_coord = idx2crd(int(blockIdx.z), shape<4>(gB_nkl));
    auto output_tile_coord = make_coord(m_coord, n_coord, _, l_coord);
```

线程块id决定了coord，是不是可以认为每个线程块负责一个tile的计算？如果每个线程块只有4个warp，也即128个thread，那很合理。

这之后经过一系列操作，貌似直接开始MMA计算了，是没有调用TMA加载吗？我们找到另一个文件`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp`，它调用了TMA的load：

```cpp
    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      Warp3 = 3
    };
```

是把不同的CTA分成了Producer和Consumer吗，这个目前还没确定，但后面我看到Producer负责load数据，Consumer负责调用mma运算。

```cpp
    if (warp_group_role == WarpGroupRole::Producer) {
      if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        collective_mainloop.load(
          params.mainloop,
          mainloop_pipeline,
          mainloop_pipe_producer_state,
          load_inputs,
          blk_coord,
          k_tile_iter, k_tile_count,
          lane_idx,
          block_rank_in_cluster,
          shared_storage.tensors.mainloop
        );
        // ...
      }
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
// 进入include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp（定位不到，可能是这个）
////////////////////////////////////////////////////////////////////////////////////////////////////

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
        // ...
    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      // ...

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

// 好几个copy，应该是下面这个，在include/cute/algorithm/copy.hpp
template <class... CT_Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...> const& atom,  // Copy_Traits may or may not have the memory barrier in it already
     Tensor<SrcEngine, SrcLayout>                 const& src,
     Tensor<DstEngine, DstLayout>                      & dst)
{
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;
  static_assert(sizeof_bits<SrcType>::value == sizeof_bits<DstType>::value);
  static_assert((is_gmem<SrcEngine>::value && is_smem<DstEngine>::value) ||
                (is_smem<SrcEngine>::value && is_gmem<DstEngine>::value),
                "Bulk Copy only supports gmem -> smem or smem -> gmem movement.");
  // G2S or S2G dispatch
  using BULK_COPY_OP = conditional_t<is_gmem<SrcEngine>::value,
                                     SM90_BULK_COPY_G2S,
                                     SM90_BULK_COPY_S2G>;

  // Find the common subtensor of src and dst
  auto tiler = max_common_layout(src, dst);
  constexpr int vec_elem = decltype(size(tiler))::value;
  constexpr int vec_bits = vec_elem * sizeof_bits_v<SrcType>;
  static_assert(vec_bits >= 128, "Expected at least 128-bits for BLKCP");

  // Construct a new concrete Atom of the vector size
  using BulkAtom = Copy_Atom<Copy_Traits<BULK_COPY_OP, Int<vec_bits>, CT_Args...>, SrcType>;
  auto bulk_atom = apply(atom.opargs_, [](auto const&... args) { return BulkAtom{args...}; });

#if 0
  if (thread0()) {
    print("copy blkcp -- found a max_common_layout of "); print(tiler); print("\n");
    print("   "); print(src); print("\n");
    print("   "); print(dst); print("\n");
  }
#endif

  return copy(bulk_atom, logical_divide(src, tiler), logical_divide(dst, tiler));
}

// 上面的copy又定位到

template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  return copy_if(copy_atom, TrivialPredTensor{}, src, dst);
}

template <class... CopyArgs,
          class PredTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(Copy_Atom<CopyArgs...>       const& copy_atom,
        PredTensor                   const& pred,      // (Rest...)
        Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
        Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");
  if constexpr (SrcLayout::rank == 1) {   // Dispatch the copy
    copy_atom.call(src, dst);
  } else {                                // Loop over all but the first mode
    constexpr int R = SrcLayout::rank;
    Tensor src_v = group_modes<1,R>(src);
    Tensor dst_v = group_modes<1,R>(dst);
    CUTE_UNROLL
    for (int i = 0; i < size<1>(src_v); ++i) {
      // If copy traits can be transformed with a predicate value, do it, otherwise branch here
      if constexpr (detail::has_with_bool<Copy_Atom<CopyArgs...>>) {
        copy_atom.with(pred(i)).call(src_v(_,i), dst_v(_,i));
      } else {
        if (pred(i)) {
          copy_atom.call(src_v(_,i), dst_v(_,i));
          // 这里的copy_atom存在于include/cute/atom/copy_traits_sm90_tma.hpp
        }
      }
    }
  }
}

  //
  // Tensor call interfaces
  //

  // Check and call instruction, or recurse
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>      & dst) const
  {
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // Dispatch to unpack to execute instruction
      return copy_unpack(*this, src, dst);
    } else
    if constexpr (is_tuple<decltype(shape(src))>::value &&
                  is_tuple<decltype(shape(dst))>::value) {
      // If the size of the src/dst doesn't match the instruction,
      //   recurse this rank-1 layout by peeling off the mode
      //   ((A,B,C,...)) -> (A,B,C,...)
      return copy(*this, tensor<0>(src), tensor<0>(dst));
    } else {
      static_assert(dependent_false<SEngine>, "No instruction match and no recursion possible.");
    }
  }
```

到了这一步，call函数中的copy_unpack终于定位到了`include/cute/atom/copy_traits_sm90_tma.hpp`或`include/cute/atom/copy_traits_sm90_im2col.hpp`里的copy_unpack或copy函数，进而会调用`include/cute/arch/copy_sm90_tma.hpp`里的copy函数，后者直接调用了ptx。

值得注意的是，在copy_unpack中可能调用`detail::explode_tuple`，会把tuple展开成多个参数，调用`CopyOp::copy()`函数。CopyOp在`include/cute/atom/copy_traits_sm90_tma.hpp`中有说明：

> @param CopyOp The target copy operation: SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST, SM90_TMA_STORE

当然根据看代码，CopyOp可以指示更多的Op，甚至包括调用了`cp.async.bulk.prefetch.tensor`。其实，所有的CopyOp定义都在`include/cute/arch/copy_sm90_tma.hpp`，就是这个文件里的每个struct，其下包含了`copy()`函数，这个函数调用了ptx指令。



还可以看看/test/unit/cute/hopper/bulk_load.cu
