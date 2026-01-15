## 5.2. [类型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#types "Permalink to this headline")

### 5.2.1. [基本类型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fundamental-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fundamental-types "Permalink to this headline")

在 PTX 中，fundamental types 反映目标架构原生支持的数据类型。fundamental type 同时指定 basic type 和 size。寄存器变量始终是 fundamental type，指令在这些类型上操作。变量定义与指令类型使用同一套 type-size specifiers，因此名称刻意简短。

[Table 8](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fundamental-types-fundamental-type-specifiers) 列出每个 basic type 的 fundamental type specifiers:

Table 8 Fundamental Type Specifiers[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fundamental-types-fundamental-type-specifiers "Permalink to this table")
|基本类型 (Basic Type)|Fundamental Type Specifiers|
|---|---|
|有符号整数|`.s8`, `.s16`, `.s32`, `.s64`|
|无符号整数|`.u8`, `.u16`, `.u32`, `.u64`|
|浮点|`.f16`, `.f16x2`, `.f32`, `.f64`|
|位 (untyped)|`.b8`, `.b16`, `.b32`, `.b64`, `.b128`|
|谓词|`.pred`|

大多数指令都有一个或多个 type specifiers，用于完整指定指令行为。操作数类型和大小会与指令类型进行兼容性检查。

当两个 fundamental types 具有相同的 basic type 且大小相同，它们是兼容的。有符号与无符号整数类型在大小相同时也兼容。bit-size type 与任意具有相同大小的 fundamental type 兼容。

原则上，所有变量（除谓词外）都可以仅使用 bit-size types 来声明，但带类型的变量能提升程序可读性，并允许更好的操作数类型检查。

### 5.2.2. [子字长尺寸的受限使用](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#restricted-use-of-sub-word-sizes)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#restricted-use-of-sub-word-sizes "Permalink to this headline")

`.u8`、`.s8` 和 `.b8` 指令类型仅限于 `ld`、`st` 和 `cvt` 指令。`.f16` 浮点类型只允许用于与 `.f32`、`.f64` 类型之间的转换、半精度浮点指令以及纹理取指指令。`.f16x2` 浮点类型只允许用于半精度浮点算术指令和纹理取指指令。

为方便起见，`ld`、`st` 和 `cvt` 指令允许源和目标数据操作数宽于 instruction-type size，从而可以使用常规宽度寄存器来加载、存储和转换窄值。例如，8-bit 或 16-bit 的值在被加载、存储或转换为其他类型和大小时，可直接保存在 32-bit 或 64-bit 寄存器中。

### 5.2.3. [替代浮点数据格式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-data-formats)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-data-formats "Permalink to this headline")

PTX 支持的 fundamental floating-point types 具有隐式位表示，用于指示指数 (exponent) 与尾数 (mantissa) 的位数。例如，`.f16` 类型表示指数占 5 bit、尾数占 10 bit。除了 fundamental types 采用的浮点表示外，PTX 还允许以下替代浮点数据格式:

`bf16` 数据格式:

该数据格式是 16-bit 浮点格式，指数 8 bit、尾数 7 bit。包含 `bf16` 数据的寄存器变量必须声明为 `.b16` 类型。

`e4m3` 数据格式:

该数据格式是 8-bit 浮点格式，指数 4 bit、尾数 3 bit。`e4m3` 编码不支持 infinity，`NaN` 值限制为 `0x7f` 和 `0xff`。包含 `e4m3` 值的寄存器变量必须使用 bit-size type 声明。

`e5m2` 数据格式:

该数据格式是 8-bit 浮点格式，指数 5 bit、尾数 2 bit。包含 `e5m2` 值的寄存器变量必须使用 bit-size type 声明。

`tf32` 数据格式:

该数据格式是矩阵乘加 (matrix multiply-and-accumulate) 指令支持的特殊 32-bit 浮点格式，范围与 `.f32` 相同且精度降低 (>=10 bits)。`tf32` 格式的内部布局由实现定义。PTX 便于将单精度 `.f32` 类型转换为 `tf32` 格式。包含 `tf32` 数据的寄存器变量必须声明为 `.b32` 类型。

`e2m1` 数据格式:

该数据格式是 4-bit 浮点格式，指数 2 bit、尾数 1 bit。`e2m1` 编码不支持 infinity 和 `NaN`。`e2m1` 值必须以 `e2m1x2` 指定的打包格式使用。包含两个 `e2m1` 值的寄存器变量必须声明为 `.b8` 类型。

`e2m3` 数据格式:

该数据格式是 6-bit 浮点格式，指数 2 bit、尾数 3 bit。`e2m3` 编码不支持 infinity 和 `NaN`。`e2m3` 值必须以 `e2m3x2` 指定的打包格式使用。包含两个 `e2m3` 值的寄存器变量必须声明为 `.b16` 类型，其中每个 `.b8` 元素包含 6-bit 浮点值，并在 2 个 MSB 位补零。

`e3m2` 数据格式:

该数据格式是 6-bit 浮点格式，指数 3 bit、尾数 2 bit。`e3m2` 编码不支持 infinity 和 `NaN`。`e3m2` 值必须以 `e3m2x2` 指定的打包格式使用。包含两个 `e3m2` 值的寄存器变量必须声明为 `.b16` 类型，其中每个 `.b8` 元素包含 6-bit 浮点值，并在 2 个 MSB 位补零。

`ue8m0` 数据格式:

该数据格式是 8-bit 无符号浮点格式，指数 8 bit、尾数 0 bit。`ue8m0` 编码不支持 infinity。`NaN` 值限制为 `0xff`。`ue8m0` 值必须以 `ue8m0x2` 指定的打包格式使用。包含两个 `ue8m0` 值的寄存器变量必须声明为 `.b16` 类型。

`ue4m3` 数据格式:

该数据格式是 7-bit 无符号浮点格式，指数 4 bit、尾数 3 bit。`ue4m3` 编码不支持 infinity。`NaN` 值限制为 `0x7f`。包含单个 `ue4m3` 值的寄存器变量必须声明为 `.b8` 类型，且最高 MSB 位补零。

替代数据格式不能作为 fundamental types 使用。它们只能在某些指令中作为源或目标格式使用。

### 5.2.4. [定点数据格式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fixed-point-data-formats)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fixed-point-data-formats "Permalink to this headline")

PTX 支持以下 fixed-point 数据格式:

`s2f6` 数据格式:

该数据格式是 8-bit 有符号二进制补码整数，包含 2 位符号整数位和 6 位小数位，形式为 **xx.xxxxxx**。`s2f6` 编码不支持 infinity 和 `NaN`。

`s2f6` value = s8 value * 2^(-6) 正最大表示 = 01.111111 = 127 * 2^(-6) = 1.984375 负最大表示 = 10.000000 = -128 * 2^(-6) = -2.0

### 5.2.5. [打包数据类型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-data-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-data-types "Permalink to this headline")

某些 PTX 指令并行地对两组或多组输入进行操作，并产生两组或多组输出。这类指令可以使用打包格式存储的数据。PTX 支持将相同标量数据类型的两个或四个值打包为一个更大的值。该打包值被视为一个 packed data type 的值。本节描述 PTX 支持的 packed data types。

#### 5.2.5.1. [打包浮点数据类型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-floating-point-data-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-floating-point-data-types "Permalink to this headline")

PTX 支持多种 packed floating point data types 变体。其中只有 `.f16x2` 支持作为 fundamental type，其他类型不能作为 fundamental types 使用，仅在某些指令中作为 instruction types 支持。当使用这些非 fundamental types 的指令时，操作数数据变量必须为合适大小的 bit type。例如，当指令类型为 `.bf16x2` 时，所有操作数变量都必须是 `.b32` 类型。[Table 9](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operand-types-for-packed-floating-point-instruction-type) 描述了 PTX 中 packed floating point data types 的各种变体。

Table 9 Operand types for packed floating point instruction type.[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operand-types-for-packed-floating-point-instruction-type "Permalink to this table")
|打包浮点类型 (Packed floating point type)|打包格式包含的元素数量 (Number of elements contained in a packed format)|每个元素的类型 (Type of each element)|声明时使用的寄存器变量类型 (Register variable type to be used in the declaration)|
|---|---|---|---|
|`.f16x2`|两个|`.f16`|`.f16x2` or `.b32`|
|`.f32x2`|`.f32`|`.b64`|
|`.bf16x2`|`.bf16`|`.b32`|
|`.e4m3x2`|`.e4m3`|`.b16`|
|`.e5m2x2`|`.e5m2`|
|`.e2m3x2`|`.e2m3`|
|`.e3m2x2`|`.e3m2`|
|`.ue8m0x2`|`.ue8m0`|
|`.s2f6x2`|`.s2f6`|
|`.e2m1x2`|`.e2m1`|`.b8`|
|`.e4m3x4`|四个|`.e4m3`|`.b32`|
|`.e5m2x4`|`.e5m2`|
|`.e2m3x4`|`.e2m3`|
|`.e3m2x4`|`.e3m2`|
|`.e2m1x4`|`.e2m1`|`.b16`|

#### 5.2.5.2. [打包整数数据类型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-integer-data-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-integer-data-types "Permalink to this headline")

PTX 支持两种 packed integer data types: `.u16x2` 和 `.s16x2`。这种 packed data type 由两个 `.u16` 或 `.s16` 值组成。包含 `.u16x2` 或 `.s16x2` 数据的寄存器变量必须声明为 `.b32` 类型。packed integer data types 不能作为 fundamental types 使用，仅在某些指令中作为 instruction types 支持。

#### 5.2.5.3. [打包定点数据类型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-fixed-point-data-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#packed-fixed-point-data-types "Permalink to this headline")

PTX 支持 `.s2f6x2` packed fixed-point data type，由两个 `.s2f6` 打包定点值组成。包含 `.s2f6x2` 值的寄存器变量必须声明为 `.b16` 类型。packed fixed-point data type 不能作为 fundamental type 使用，并且仅作为 instruction type 支持。
