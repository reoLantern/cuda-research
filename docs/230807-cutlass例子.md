# turing_tensorop_conv2dfprop.cu

[数制和Array的官方文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md)

阅读源码，解释该例子的逻辑。

## 调用mma.sync

首先，在本文件turing_tensorop_conv2dfprop.cu中，通过以下代码进行矩阵运算。

```c++
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
Result profile_convolution(Options const &options) {
    ImplicitGemm implicit_gemm_op;
    // ...
    result.status = implicit_gemm_op();
    // ...
    return result;
}
int main(int argc, char const **args){
    // ...
    Result result = profile_convolution(options);
    // ...
    return 0;
}
```

考虑`implicit_gemm_op()`的实现。在于ImplicitGemmConvolution的定义，在cutlass/conv/device/implicit_gemm_convolution.h中（节选）：

```c++
template<typename ImplicitGemmKernel_>
class ImplicitGemmConvolution {
public:
  using UnderlyingKernel = ImplicitGemmKernel_;
  using ElementA = typename UnderlyingKernel::ElementA;
  using LayoutA = typename UnderlyingKernel::LayoutA;
  using ElementB = typename UnderlyingKernel::ElementB;
  using LayoutB = typename UnderlyingKernel::LayoutB;
  using ElementC = typename UnderlyingKernel::ElementC;
  using LayoutC = typename UnderlyingKernel::LayoutC;

private:
  typename UnderlyingKernel::Params params_;

public:
  Status run(cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));

    cutlass::Kernel<UnderlyingKernel><<<grid, block, smem_size, stream>>>(params_);

    cudaError_t result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
}
```

而语句`cutlass::Kernel<UnderlyingKernel><<<grid, block, smem_size, stream>>>(params_);`，UnderlyingKernel实际为Conv2dFpropKernel。后者实际上是DefaultConv2dFprop中的Kernel，从cutlass/conv/kernel/default_conv2d_fprop.h的第838行开始：

```c++
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};
```

实际上执行的核函数为`DefaultConv2dFprop::Kernel()`，其中Kernel来自cutlass/conv/kernel/ImplicitGemmConvolution。在它的定义中，存在如下语句：

```c++
template <
  typename Mma_,                                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,                             ///! Epilogue
  // ...
>
struct ImplicitGemmConvolution {
  using Mma = Mma_;
  using Epilogue = Epilogue_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  // ...

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    // ...

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    // Compute threadblock-scoped matrix multiply-add
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, params.gemm_k_iterations_per_channel);

    // ...
  }
}
```

可以看到，实际运行的是`Mma()`，也即`threadblock::ImplicitGemmPipelined()`，同时将iterator_A等参数传入函数。ImplicitGemmPipelined的定义如下：

```c++
/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Iterates over tiles of A operand in global memory 
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of A operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of B operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_,
  /// Transformation applied to A operand
  typename TransformA_ = NumericArrayConverter<
    typename SmemIteratorA_::Element, 
    typename IteratorA_::Element, 
    IteratorA_::Fragment::kElements>,
  ///
  /// Transformation applied to A operand
  typename TransformB_ = NumericArrayConverter<
    typename SmemIteratorB_::Element, 
    typename IteratorB_::Element, 
    IteratorB_::Fragment::kElements>,
  /// Used for partial specialization
  typename Enable = bool
>
class ImplicitGemmPipelined : public gemm::threadblock::MmaBase<Shape_, Policy_, 2> {
  // ...

public:
  using Policy = Policy_;           ///< Policy describing tuning details
  /// Warp-level Mma
  using Operator = typename Policy::Operator;

private:
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;

public:
  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum,                       ///< source accumulator tile
    int gemm_k_iterations_per_channel = 0,             ///< number of iterations per channel
    TransformA transform_A = TransformA(),            ///< transformation applied to A fragment
    TransformB transform_B = TransformB()) {          ///< transformation applied to B fragment
    // ...

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    // The last kblock is loaded in the prolog
    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    ++iterator_A;
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A(tb_frag_A));
    this->smem_iterator_B_.store(transform_B(tb_frag_B));

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A(tb_frag_A));

          this->smem_iterator_B_.store(transform_B(tb_frag_B));

          __syncthreads();
          
          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;

          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          }
          else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {

          iterator_A.load(tb_frag_A);
          iterator_B.load(tb_frag_B);
    
          ++iterator_A;
          ++iterator_B;
        }

        warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }

  }

  // ...
}
```

因此，实际调用了`warp_mma(accum, warp_frag_A[warp_mma_k % 2], warp_frag_B[warp_mma_k % 2], accum);`来执行一次tensor运算。我的理解是，核函数内部负责管理多个tile的迭代，warp_frag_A的类型WarpFragmentA、warp_mma函数的类型Operator最终都来自DefaultConv2dFprop的MmaPolicy，而后者又来自MmaCore，即：

```c++
using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
    ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
    2, MathOperatorTag>;
```

所谓DefaultMmaCore，是Arch-specified的，每一个版本的架构都有相应的定义。本例使用的版本是sm_75，对应的DefaultMmaCore定义如下：

```c++
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by MMA
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2, Operator_
                      > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement 
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    0,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    1,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};
```

这里可以看到，Policy的定义来自MmaTensorOp，实际上刚刚提到的warp_frag_A数据和warp_mma函数都包含在这个MmaTensorOp中。后者的定义如下：

```c++
/// Partial specialization for m-by-n-by-kgroup
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Data type of B elements
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Operator describing the tensor operation
    typename Operator_,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp {
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<InstructionShape_, 32, ElementA,
                         cutlass::layout::RowMajor, ElementB,
                         cutlass::layout::ColumnMajor, ElementC,
                         cutlass::layout::RowMajor, Operator_>,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};
```

到这里就非常明了了，所有的类型定义最后都落实到`cutlass::arch::Mma`上。它也是Arch-specified，同时，还能够自动匹配数据类型。例如，int4b_t对应的`cutlass::arch::Mma`为：

```c++
/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);
  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};
```

这里，定义了warp_frag_A的类型为：`using FragmentA = Array<int4b_t, 8>;`；定义运算函数为`void operator();`，使用内联汇编进行一次tensor core运算。

## 数据传递

现在，我们已经知道了运算逻辑，接下来弄清楚矩阵数据具体是如何组织的就可以了。实际上，笔者之前的文档中提到过cutlass对于自定义类型cutlass::int4b_t的定义。它使用uint8_t存储2个packed的4-bit整数；同时cutlass定义了类型Array，在上面的mma函数中，Fragment是`Array<int4b_t, 8>`，每个线程负责为mma计算提供8个packed在单个32-bit寄存器中的4-bit整数。

我们从mma操作的计算开始回溯数据的来源。

```c++
/// Fragment of operand A loaded from global memory
using FragmentA = typename IteratorA::Fragment;
FragmentA tb_frag_A;
iterator_A.load(tb_frag_A);

this->smem_iterator_A_.store(transform_A(tb_frag_A));

using WarpFragmentA = typename Operator::FragmentA;
WarpFragmentA warp_frag_A[2];
this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);

// threadblock/mma_base.h:
typename Operator::IteratorA warp_tile_iterator_A_;
using Operator = typename Policy::Operator;

// arch/mma_sm75.h:
using Operator = OpMultiplyAdd;
```

其中，tb_frag_A的类型是IteratorA::Fragment，它是`Array<int4b_t, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>`。用于为每个线程指定特定位置的存放在global mem中的矩阵数据。

warp_frag_A的类型是Policy::Operator::FragmentA，它是`Array<int4b_t, 8>`。负责单次tensor core运算中从shared mem加载到寄存器的数据。

考虑这个`Array<int4b_t, 8>`，在Array的定义中，`sizeof_bits<int4b_t>=4`，因此（以下是Array内部的定义）有kSizeBits=32，Storage=uint32_t，Element=int4b_t。因此有kElementsPerStoredItem=8，kStorageElements=1，因此，实际上`Array<int4b_t, 8>`存储的数据为`uint32_t storage[1];`。

考察`iterator_A.load(tb_frag_A);`语句，它用到了iterator_A的自定义函数load()，输入参数是一个自定义类型的Array。iterator的类型是cutlass::conv::threadblock::TileIterator，在default_conv2d_fprop.h被调用。它的调用和定义如下：

```c++
  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;
```

进一步，TileIterator的定义如下：

```c++
template <typename TileAccessIterator_>
class TileIterator {
public:
  using TileAccessIterator = TileAccessIterator_;

  using Shape = typename TileAccessIterator::Shape;
  using Element = typename TileAccessIterator::Element;
  using Layout = typename TileAccessIterator::Layout;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = typename TileAccessIterator::ThreadMap;
  using AccessType = typename TileAccessIterator::AccessType;
  using TensorRef = typename TileAccessIterator::TensorRef;
  using Index = typename TileAccessIterator::Index;
  using LongIndex = typename TileAccessIterator::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = TileAccessIterator::kIteratorAlgorithm;
  static StrideSupport const kStrideSupport = TileAccessIterator::kStrideSupport;
  using Params = typename TileAccessIterator::Params;
  static int const kConvDim = TileAccessIterator::kConvDim;
  using ConvProblemSize = typename TileAccessIterator::ConvProblemSize;
  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
    Element, 
    ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

private:

  /// Internal state
  TileAccessIterator tile_access_iterator_;

public:

  /// Constructor
  CUTLASS_HOST_DEVICE
  TileIterator(
    Params const &params,
    ConvProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    tile_access_iterator_(params, problem_size, ptr, thread_idx, threadblock_offset) { }

  // ...

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    frag.clear();
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          cutlass::arch::global_load<
            AccessType,
            sizeof(AccessType)
          >(
            frag_ptr[idx],
            tile_access_iterator_.get() + pointer_offset,
            tile_access_iterator_.valid()
          );
  
          ++tile_access_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    tile_access_iterator_.set_iteration_index(0);
    load_with_pointer_offset(frag, 0);
  }

};
```

它使用`cutlass::arch::global_load()`函数将全局内存中的矩阵按照特定的顺序、布局存放到已定义的Array（即Fragment）。pointer_offset的类型是LayoutInputA::Index=int32_t。

其中，frag的类型是`Array<int4b_t, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>`，会进行kCount次迭代，每次加载的AccessType是`using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;`。这个AlignedArray是自定义类型Array的子类，表示一个特定类型和大小的Array。由于此处AccessType的类型是`Array<int4b_t, ThreadMap_::kElementsPerAccess>`，而`kElementsPerAccess = kAccessSizeInBits / sizeof_bits<ElementA>::value = 128 / 4 = 32`，即`AccessType=Array<int4b_t, 32>`，因此sizeof(AccessType)=16。对应的global_load函数有两种：

```c++
// The redundant mov PTX instruction is used to enforce the compiler to
// keep the initializing code before ld.global
template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#else
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#endif
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};
```

可以看到，它使用内联汇编，data就是一个`Array<int4b_t, 32>`，可以直接转换成4个32bit，ptr是相应矩阵数据的指针。函数把以int4b_t格式存储的矩阵，加载到Array中。ptr即是`tile_access_iterator_.get()`，这个函数的定义如下，在conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h中：

```c++
  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {

    TensorCoord coord = at();
    LongIndex offset = params_.layout(coord);
    
    AccessType const *ptr = reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);

    return ptr;
  }
```

AccessType是Array，pointer_的type是cutlass::int4b_t。因此，cutlass通过这种指针赋值的方式，实现数据的传递。实际上，这些4-bit整数一直都是packed的。

回顾前面：

```c++
AccessType const *ptr = reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);
```

上面的语句中，Element为int4b_t，所以`offset * sizeof_bits<Element>::value / 8`实际为`offset * 4 / 8`。offset来自：

```c++
  /// Returns the offset of a coordinate (n, h, w, c) in linear memory. 
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.c() + 
      LongIndex(stride_[0] * coord.w()) + 
      LongIndex(stride_[1] * coord.h()) +
      LongIndex(stride_[2] * coord.n());
  }
```

上述语句的`pointer_`在cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h中定义如下：

```c++
Element const *ptr;     // Element=int4b_t
const char *pointer_ = reinterpret_cast<char const *>(ptr);
```

也即`pointer_`是指向uint8_t的指针。

## 数据索引和初始化

既然`cutlass::int4b_t`将每两个4-bit整数pack进1个`uint8_t`中，就需要知道cutlass如何实现数据的索引。这可以通过观察数据初始化的代码了解这一过程。

实际上，由于我们用指针赋值的方式传递数据，数据类型本身并不重要，int4b_t用uint8_t存数据、还是用uint16_t存数据不重要。cutlass采用`TensorRef<ElementType>`来定位到任意数据类型的任意一个元素的位置。我们可以简单认为，每一个`TensorRef`唯一对应了一个数据指针和一个offset，从而唯一对应了一个任意`ElementType`类型的数据。

当数据类型是`int4b_t`时，`TensorRef`会自动调用`SubbyteReference`，功能是一样的。根据后者的定义：

```c++
template <
  typename Element_,              /// CUTLASS numeric element type.
  typename Storage_ = uint16_t
>
class SubbyteReference {
public:

  using Element = Element_;
  using Storage = Storage_;
  using StoragePointer = Storage *;

private:

  ///! Number of elements per storage vector
  int const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;  // = 16 / 4 = 4

  ///! Bit mask 
  Storage const kMask =     // = '0000000000001111'
    ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? 
      (Storage(1) << sizeof_bits<Element>::value) - Storage(1) :
      ~Storage(0));

private:

  /// Pointer to array containing element
  StoragePointer ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

public:

  /// Constructor
  CUTLASS_HOST_DEVICE
  SubbyteReference(
    Element *ptr,           /// pointer to memory
    int64_t offset          /// logical offset in units of Element
  ): 
    ptr_(reinterpret_cast<StoragePointer>(ptr)),
    offset_(0) {

    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);
  }
};
```

有一个数据指针`ptr_`，还有一个偏移量`offset`，这个offset就是第几个4-bit整数的意思。当创建一个新的 `TensorRef<cutlass::int4b_t, layout>`并使用索引访问元素时，程序可以正确地访问到每一个4-bit整数的reference。

此外，在初始化 HostTensor 时，在 tensor_fill.h 中，会将一个 `cutlass::int4b_t` 赋值给一个 `SubbyteReference`，这将确保 `int4b_t` 数据的低 4 位正确分配给  `SubbyteReference` 代表的 4 比特。
