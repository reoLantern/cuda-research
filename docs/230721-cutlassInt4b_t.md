## 类型定义

cutlass定义了类int4b_t来表示4bit的整数，定义如下。

```c++
/// 4-bit signed integer type
template <int Bits, bool Signed = true>
struct integer_subbyte {

  /// Number of bits
  static int const kBits = Bits;

  /// Whether type is signed
  static bool const kSigned = Signed;

  /// External type
  using T = typename platform::conditional<kSigned, int, unsigned>::type;

  /// Storage type
  using Storage = uint8_t;

  /// Bitmask used to truncate from larger integers
  static Storage const kMask = Storage((1 << kBits) - 1);

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  integer_subbyte() = default;

  /// Conversion from integer type
  CUTLASS_HOST_DEVICE
  integer_subbyte(int value)
      : storage(reinterpret_cast<Storage const &>(value) & kMask) {}

  CUTLASS_HOST_DEVICE
  integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const &>(value) & kMask) {}

  CUTLASS_HOST_DEVICE
  integer_subbyte(double value) {
    T tmp = static_cast<T>(value);
    storage = Storage(reinterpret_cast<unsigned const &>(tmp) & kMask);
  }

  ///
  CUTLASS_HOST_DEVICE
  operator T() const {
    if (kSigned) {
      // Sign extend
      if (storage & Storage(1 << (kBits - 1))) {
        return T(storage) | ~T(kMask);
      }
    }
    return T(storage);
  }

  /// Equality
  CUTLASS_HOST_DEVICE
  bool operator==(integer_subbyte const &rhs) const {
    return storage == rhs.storage;
  }

  /// Inequality
  CUTLASS_HOST_DEVICE
  bool operator!=(integer_subbyte const &rhs) const {
    return storage != rhs.storage;
  }

  /// Less than or equal
  CUTLASS_HOST_DEVICE
  bool operator<=(integer_subbyte const &rhs) const {
    if (kSigned) {
      if (storage & (1 << (kBits - 1))) {
        return !(rhs.storage < storage);
      }
    }
    return storage < rhs.storage;
  }

  /// Less than
  CUTLASS_HOST_DEVICE
  bool operator<(integer_subbyte const &rhs) const {
    if (kSigned) {
      if (storage & (1 << (kBits - 1))) {
        return !(rhs.storage <= storage);
      }
    }
    return storage < rhs.storage;
  }

  /// Greater than or equal
  CUTLASS_HOST_DEVICE
  bool operator>=(integer_subbyte const &rhs) const {
    return !(*this < rhs);
  }

  /// Greater than
  CUTLASS_HOST_DEVICE
  bool operator>(integer_subbyte const &rhs) const {
    return !(*this <= rhs);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////


/// 1-bit Unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit Integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit Unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 4-bit Integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit Unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an element in bits - specialized for uint1b_t
template <>
struct sizeof_bits<uint1b_t> {
  static int const value = 1;
};

/// Defines the size of an element in bits - specialized for int2b_t
template <>
struct sizeof_bits<int2b_t> {
  static int const value = 2;
};

/// Defines the size of an element in bits - specialized for uint2b_t
template <>
struct sizeof_bits<uint2b_t> {
  static int const value = 2;
};

/// Defines the size of an element in bits - specialized for int4b_t
template <>
struct sizeof_bits<int4b_t> {
  static int const value = 4;
};

/// Defines the size of an element in bits - specialized for uint4b_t
template <>
struct sizeof_bits<uint4b_t> {
  static int const value = 4;
};
```

它的基本含义是使用uint8_t的低四位存储这个4bit int，同时定义一个mask来标志只有低四位有效。

在mma_sm75.h中，实现了4bit整数的mma运算。这个函数在cutlass/conv/device/implicit_gemm_convolution.h中作为kernel函数被调用。它的实现如下：

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

其中，`using FragmentA = Array<int4b_t, 8>;`语句值得注意。Array是自定义类型，它的模板参数包含自定义类型int4b_t，尽管int4b_t以uint8_t保存1个4bit整数，但是Array函数支持了将这8个int4b_t的每个数据的有效4bit提取出来，打包在1个32bit内存空间中。Array类通过reference()函数实现上述功能，其代码如下。

```c++
/// Statically sized array for any data type
template <
  typename T,
  int N
>
class Array<T, N, false> {
public:

  static int const kSizeBits = sizeof_bits<T>::value * N;

  /// Storage type
  using Storage = typename platform::conditional<
    ((kSizeBits % 32) != 0),
    typename platform::conditional<
      ((kSizeBits % 16) != 0),
      uint8_t,
      uint16_t
    >::type,
    uint32_t
  >::type;

  /// Element type
  using Element = T;

  /// Number of logical elements per stored object
  static int const kElementsPerStoredItem = int(sizeof(Storage) * 8) / sizeof_bits<T>::value;

  /// Number of storage elements
  static size_t const kStorageElements = N / kElementsPerStoredItem;

  /// Number of logical elements
  static size_t const kElements = N;

  /// Bitmask for covering one item
  static Storage const kMask = ((Storage(1) << sizeof_bits<T>::value) - 1);

  //
  // C++ standard members with pointer types removed
  //

  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type *pointer;
  typedef value_type const *const_pointer;

  //
  // References
  //

  /// Reference object inserts or extracts sub-byte items
  class reference {
    /// Pointer to storage element
    Storage *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    reference(): ptr_(nullptr), idx_(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    reference(Storage *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    /// Assignment
    CUTLASS_HOST_DEVICE
    reference &operator=(T x) {
      Storage item = (reinterpret_cast<Storage const &>(x) & kMask);

      Storage kUpdateMask = Storage(~(kMask << (idx_ * sizeof_bits<T>::value)));
      *ptr_ = Storage(((*ptr_ & kUpdateMask) | (item << idx_ * sizeof_bits<T>::value)));

      return *this;
    }

    CUTLASS_HOST_DEVICE
    T get() const {
      Storage item = Storage((*ptr_ >> (idx_ * sizeof_bits<T>::value)) & kMask);
      return reinterpret_cast<T const &>(item);
    }

    /// Extract
    CUTLASS_HOST_DEVICE
    operator T() const {
      return get();
    }

    /// Explicit cast to int
    CUTLASS_HOST_DEVICE
    explicit operator int() const {
      return int(get());
    }

    /// Explicit cast to float
    CUTLASS_HOST_DEVICE
    explicit operator float() const {
      return float(get());
    }
  };

  /// Reference object extracts sub-byte items
  class const_reference {

    /// Pointer to storage element
    Storage const *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    const_reference(): ptr_(nullptr), idx_(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    const_reference(Storage const *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    CUTLASS_HOST_DEVICE
    const T get() const {
      Storage item = (*ptr_ >> (idx_ * sizeof_bits<T>::value)) & kMask;
      return reinterpret_cast<T const &>(item);
    }

    /// Extract
    CUTLASS_HOST_DEVICE
    operator T() const {
      Storage item = Storage(Storage(*ptr_ >> Storage(idx_ * sizeof_bits<T>::value)) & kMask);
      return reinterpret_cast<T const &>(item);
    }

    /// Explicit cast to int
    CUTLASS_HOST_DEVICE
    explicit operator int() const {
      return int(get());
    }

    /// Explicit cast to float
    CUTLASS_HOST_DEVICE
    explicit operator float() const {
      return float(get());
    }
  };

  //
  // Iterators
  //

  /// Bidirectional iterator over elements
  class iterator {

    /// Pointer to storage element
    Storage *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    CUTLASS_HOST_DEVICE
    iterator(): ptr_(nullptr), idx_(0) { }

    CUTLASS_HOST_DEVICE
    iterator(Storage *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    CUTLASS_HOST_DEVICE
    iterator &operator++() {
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator &operator--() {
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator operator++(int) {
      iterator ret(*this);
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    iterator operator--(int) {
      iterator ret(*this);
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    reference operator*() const {
      return reference(ptr_, idx_);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(iterator const &other) const {
      return ptr_ == other.ptr_ && idx_ == other.idx_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(iterator const &other) const {
      return !(*this == other);
    }
  };

  /// Bidirectional constant iterator over elements
  class const_iterator {

    /// Pointer to storage element
    Storage const *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    CUTLASS_HOST_DEVICE
    const_iterator(): ptr_(nullptr), idx_(0) { }

    CUTLASS_HOST_DEVICE
    const_iterator(Storage const *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    CUTLASS_HOST_DEVICE
    iterator &operator++() {
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator &operator--() {
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator operator++(int) {
      iterator ret(*this);
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    iterator operator--(int) {
      iterator ret(*this);
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_reference operator*() const {
      return const_reference(ptr_, idx_);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(iterator const &other) const {
      return ptr_ == other.ptr_ && idx_ == other.idx_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(iterator const &other) const {
      return !(*this == other);
    }
  };

  /// Bidirectional iterator over elements
  class reverse_iterator {

    /// Pointer to storage element
    Storage *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    CUTLASS_HOST_DEVICE
    reverse_iterator(): ptr_(nullptr), idx_(0) { }

    CUTLASS_HOST_DEVICE
    reverse_iterator(Storage *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }
  };

  /// Bidirectional constant iterator over elements
  class const_reverse_iterator {

    /// Pointer to storage element
    Storage const *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(): ptr_(nullptr), idx_(0) { }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(Storage const *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }
  };

private:

  /// Internal storage
  Storage storage[kStorageElements];

public:

  #if 0
  CUTLASS_HOST_DEVICE
  Array() { }

  CUTLASS_HOST_DEVICE
  Array(Array const &x) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(kStorageElements); ++i) {
      storage[i] = x.storage[i];
    }
  }
  #endif

  /// Efficient clear method
  CUTLASS_HOST_DEVICE
  void clear() {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(kStorageElements); ++i) {
      storage[i] = Storage(0);
    }
  }

  CUTLASS_HOST_DEVICE
  reference at(size_type pos) {
    return reference(storage + pos / kElementsPerStoredItem, pos % kElementsPerStoredItem);
  }

  CUTLASS_HOST_DEVICE
  const_reference at(size_type pos) const {
    return const_reference(storage + pos / kElementsPerStoredItem, pos % kElementsPerStoredItem);
  }

  CUTLASS_HOST_DEVICE
  reference operator[](size_type pos) {
    return at(pos);
  }

  CUTLASS_HOST_DEVICE
  const_reference operator[](size_type pos) const {
    return at(pos);
  }

  CUTLASS_HOST_DEVICE
  reference front() {
    return at(0);
  }

  CUTLASS_HOST_DEVICE
  const_reference front() const {
    return at(0);
  }

  CUTLASS_HOST_DEVICE
  reference back() {
    return reference(storage + kStorageElements - 1, kElementsPerStoredItem - 1);
  }

  CUTLASS_HOST_DEVICE
  const_reference back() const {
    return const_reference(storage + kStorageElements - 1, kElementsPerStoredItem - 1);
  }

  CUTLASS_HOST_DEVICE
  pointer data() {
    return reinterpret_cast<pointer>(storage);
  }

  CUTLASS_HOST_DEVICE
  const_pointer data() const {
    return reinterpret_cast<const_pointer>(storage);
  }
  
  CUTLASS_HOST_DEVICE
  Storage * raw_data() {
    return storage;
  }

  CUTLASS_HOST_DEVICE
  Storage const * raw_data() const {
    return storage;
  }


  CUTLASS_HOST_DEVICE
  constexpr bool empty() const {
    return !kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type max_size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  void fill(T const &value) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerStoredItem; ++i) {
      reference ref(storage, i);
      ref = value;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kStorageElements; ++i) {
      storage[i] = storage[0];
    }
  }

  CUTLASS_HOST_DEVICE
  iterator begin() {
    return iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  const_iterator cbegin() const {
    return const_iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  iterator end() {
    return iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  const_iterator cend() const {
    return const_iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rbegin() {
    return reverse_iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rend() {
    return reverse_iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crend() const {
    return const_reverse_iterator(storage);
  }

  //
  // Comparison operators
  //

};
```

Array将数据的打包似乎不是在kernel函数完成的，它应当在以下代码段完成：

```c++
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_c.device_ref(),
    {options.alpha, options.beta},
  };
```

即在arguments初始化时就完成了对数组的打包。数组本身是device端定义的内存空间，不过打包应当还是由cpu完成。
