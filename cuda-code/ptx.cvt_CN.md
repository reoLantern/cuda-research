#### 9.7.9.21. [数据移动与转换指令: `cvt`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt "Permalink to this headline")

`cvt`

将一个值从一种类型转换为另一种类型。

语法

```
cvt{.irnd}{.ftz}{.sat}.dtype.atype         d, a;  // integer rounding
cvt{.frnd}{.ftz}{.sat}.dtype.atype         d, a;  // fp rounding

cvt.frnd2{.relu}{.satfinite}.f16.f32       d, a;
cvt.frnd2{.relu}{.satfinite}.f16x2.f32     d, a, b;
cvt.rs{.relu}{.satfinite}.f16x2.f32        d, a, b, rbits;

cvt.frnd2{.relu}{.satfinite}.bf16.f32      d, a;
cvt.frnd2{.relu}{.satfinite}.bf16x2.f32    d, a, b;
cvt.rs{.relu}{.satfinite}.bf16x2.f32       d, a, b, rbits;

cvt.rna{.satfinite}.tf32.f32               d, a;
cvt.frnd2{.satfinite}{.relu}.tf32.f32      d, a;

cvt.rn.satfinite{.relu}.f8x2type.f32       d, a, b;
cvt.rn.satfinite{.relu}.f8x2type.fp16x2    d, a;
cvt.rn.{.relu}.f16x2.f8x2type              d, a;
cvt.rs{.relu}.satfinite.f8x4type.f32       d, {a, b, e, f}, rbits;

cvt.rn.satfinite{.relu}.f4x2type.f32        d, a, b;
cvt.rn.satfinite{.relu}.f4x2type.fp16x2type d, a;
cvt.rn{.relu}.f16x2.f4x2type                d, a;
cvt.rs{.relu}.satfinite.f4x4type.f32        d, {a, b, e, f}, rbits;

cvt.rn.satfinite{.relu}.f6x2type.f32        d, a, b;
cvt.rn.satfinite{.relu}.f6x2type.fp16x2type d, a;
cvt.rn{.relu}.f16x2.f6x2type                d, a;
cvt.rs{.relu}.satfinite.f6x4type.f32        d, {a, b, e, f}, rbits;

cvt.frnd3{.satfinite}.ue8m0x2.f32          d, a, b;
cvt.frnd3{.satfinite}.ue8m0x2.bf16x2       d, a;
cvt.rn.bf16x2.ue8m0x2                      d, a;

cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.f32      d, a, b{, scale-factor};
cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.bf16x2   d, a{, scale-factor};
cvt.rn{.satfinite}{.relu}{.scaled::n2::ue8m0}.bf16x2.s2f6x2 d, a{, scale-factor};

.irnd   = { .rni, .rzi, .rmi, .rpi };
.frnd   = { .rn,  .rz,  .rm,  .rp  };
.frnd2  = { .rn,  .rz };
.frnd3  = { .rz,  .rp };
.dtype = .atype = { .u8,   .u16, .u32, .u64,
                    .s8,   .s16, .s32, .s64,
                    .bf16, .f16, .f32, .f64 };
.f8x2type = { .e4m3x2, .e5m2x2 };
.f4x2type = { .e2m1x2 };
.f6x2type = { .e2m3x2, .e3m2x2 };
.f4x4type = { .e2m1x4 };
.f8x4type = { .e4m3x4, .e5m2x4 };
.f6x4type = { .e2m3x4, .e3m2x4 };
.fp16x2type = { .f16x2, .bf16x2 };
```

说明

在不同类型和位宽之间进行转换。

对于 `.f16x2` 和 `.bf16x2` 指令类型，两个 `.f32` 类型输入 `a` 和 `b` 会被转换为 `.f16` 或 `.bf16` 类型，并将转换后的值打包到目标寄存器 `d` 中，使得由输入 `a` 转换得到的值存储在 `d` 的高半部分，而由输入 `b` 转换得到的值存储在 `d` 的低半部分。

对于 `.f16x2` 指令类型，目标操作数 `d` 具有 `.f16x2` 或 `.b32` 类型。对于 `.bf16` 指令类型，操作数 `d` 具有 `.b16` 类型。对于 `.bf16x2` 指令类型，操作数 `d` 具有 `.b32` 类型。对于 `.tf32` 指令类型，操作数 `d` 具有 `.b32` 类型。

当转换到 `.e4m3x2`/`.e5m2x2` 数据格式时，目标操作数 `d` 具有 `.b16` 类型。当将两个 `.f32` 输入转换到 `.e4m3x2`/`.e5m2x2` 时，每个输入都会被转换为指定格式，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 转换得到的值存储在 `d` 的高 8 位，而由输入 `b` 转换得到的值存储在 `d` 的低 8 位。当将一个 `.f16x2`/`.bf16x2` 输入转换到 `.e4m3x2`/`.e5m2x2` 时，来自操作数 `a` 的每个 `.f16`/`.bf16` 输入都会被转换为指定格式。转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 的高 16 位转换得到的值存储在 `d` 的高 8 位，而由输入 `a` 的低 16 位转换得到的值存储在 `d` 的低 8 位。

当从 `.e4m3x2`/`.e5m2x2` 转换到 `.f16x2` 时，源操作数 `a` 具有 `.b16` 类型。操作数 `a` 中的每个 8-bit 输入值都会被转换为 `.f16` 类型。转换后的值被打包到目标操作数 `d` 中，使得由 `a` 的高 8 位转换得到的值存储在 `d` 的高 16 位，而由 `a` 的低 8 位转换得到的值存储在 `d` 的低 16 位。

当转换到 `.e2m1x2` 数据格式时，目标操作数 `d` 具有 `.b8` 类型。当将两个 `.f32` 输入转换到 `.e2m1x2` 时，每个输入都会被转换为指定格式，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 转换得到的值存储在 `d` 的高 4 位，而由输入 `b` 转换得到的值存储在 `d` 的低 4 位。当将一个 `.f16x2`/`.bf16x2` 输入转换到 `.e2m1x2` 时，来自操作数 `a` 的每个 `.f16`/`.bf16` 输入都会被转换为指定格式。转换后的值被打包到 `d` 中，使得由 `a` 的高 16 位得到的值存储在 `d` 的高 4 位，而由 `a` 的低 16 位得到的值存储在 `d` 的低 4 位。

当从 `.e2m1x2` 转换到 `.f16x2` 时，源操作数 `a` 具有 `.b8` 类型。操作数 `a` 中的每个 4-bit 输入值都会被转换为 `.f16` 类型。转换后的值被打包到目标操作数 `d` 中，使得由 `a` 的高 4 位转换得到的值存储在 `d` 的高 16 位，而由 `a` 的低 4 位转换得到的值存储在 `d` 的低 16 位。

当转换到 `.e2m1x4` 数据格式时，目标操作数 `d` 具有 `.b16` 类型。当将四个 `.f32` 输入转换到 `.e2m1x4` 时，每个输入都会被转换为指定格式，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a`、`b`、`e`、`f` 转换得到的值依次存储在 `d` 的高位起每 4 bit 的位置。

当转换到 `.e2m3x2`/`.e3m2x2` 数据格式时，目标操作数 `d` 具有 `.b16` 类型。当将两个 `.f32` 输入转换到 `.e2m3x2`/`.e3m2x2` 时，每个输入都会被转换为指定格式，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 转换得到的值存储在 `d` 的高 8 位并在最高 2 个 MSB 位补零，而由输入 `b` 转换得到的值存储在 `d` 的低 8 位并在最高 2 个 MSB 位补零。当将一个 `.f16x2`/`.bf16x2` 输入转换到 `.e2m3x2`/`.e3m2x2` 时，来自操作数 `a` 的每个 `.f16`/`.bf16` 输入都会被转换为指定格式。转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 的高 16 位得到的值存储在 `d` 的高 8 位并在最高 2 个 MSB 位补零，而由输入 `a` 的低 16 位得到的值存储在 `d` 的低 8 位并在最高 2 个 MSB 位补零。

当从 `.e2m3x2`/`.e3m2x2` 转换到 `.f16x2` 时，源操作数 `a` 具有 `.b16` 类型。操作数 `a` 中每个带有 2 个 MSB 位为 0 的 8-bit 输入值都会被转换为 `.f16` 类型。转换后的值被打包到目标操作数 `d` 中，使得由 `a` 的高 8 位转换得到的值存储在 `d` 的高 16 位，而由 `a` 的低 8 位转换得到的值存储在 `d` 的低 16 位。

当转换到 `.e5m2x4`/`.e4m3x4`/`.e3m2x4`/`.e2m3x4` 数据格式时，目标操作数 `d` 具有 `.b32` 类型。当将四个 `.f32` 输入转换到 `.e5m2x4`/`.e4m3x4`/`.e3m2x4`/`.e2m3x4` 时，每个输入都会被转换为指定格式，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a`、`b`、`e`、`f` 转换得到的值依次存储在 `d` 的高位起每 8 bit 的位置。对于 `.e3m2x4`/`.e2m3x4`，每个 8-bit 输出都会在最高 2 个 MSB 位补零。

当转换到 `.ue8m0x2` 数据格式时，目标操作数 `d` 具有 `.b16` 类型。当将两个 `.f32` 或两个打包的 `.bf16` 输入转换到 `.ue8m0x2` 时，每个输入都会被转换为指定格式，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 转换得到的值存储在 `d` 的高 8 位，而由输入 `b` 转换得到的值存储在 `d` 的低 8 位。

当从 `.ue8m0x2` 转换到 `.bf16x2` 时，源操作数 `a` 具有 `.b16` 类型。操作数 `a` 中的每个 8-bit 输入值都会被转换为 `.bf16` 类型。转换后的值被打包到目标操作数 `d` 中，使得由 `a` 的高 8 位转换得到的值存储在 `d` 的高 16 位，而由 `a` 的低 8 位转换得到的值存储在 `d` 的低 16 位。

当转换到 `.s2f6x2` 数据格式时，目标操作数 `d` 具有 `.b16` 类型。当将两个 `.f32` 输入转换到 `.s2f6x2` 时，每个输入都会被转换为 `.s2f6` 值，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 转换得到的值存储在 `d` 的高 8 位，而由输入 `b` 转换得到的值存储在 `d` 的低 8 位。当从 `.bf16x2` 转换时，打包在操作数 `a` 中的每个输入都会被转换为 `.s2f6` 值，转换后的值被打包到目标操作数 `d` 中，使得由输入 `a` 的高 8 位转换得到的值存储在操作数 `d` 的高 8 位，而由输入 `a` 的低 8 位转换得到的值存储在操作数 `d` 的低 8 位。可选操作数 `scale-factor` 的类型为 `.b16`，用于存储两个打包的 `.ue8m0` 类型缩放因子。对于向下转换，输入会先除以 `scale-factor` 再进行转换。输入 `a`/来自 `a` 高 16 位的 `.bf16` 输入的缩放因子存储在操作数 `scale-factor` 的高 8 位，输入 `b`/来自 `a` 低 16 位的 `.bf16` 输入的缩放因子存储在操作数 `scale-factor` 的低 8 位。如果未显式指定，`scale-factor` 的值假定为 `0x7f7f`，即两个输入缩放因子均为值 `1`。

当从 `.s2f6x2` 转换到 `.bf16x2` 时，源操作数 `a` 具有 `.b16` 类型。操作数 `a` 中的每个 8-bit 输入值都会被转换为 `.bf16` 类型。转换后的值被打包到目标操作数 `d` 中，使得由 `a` 的高 8 位转换得到的值存储在 `d` 的高 16 位，而由 `a` 的低 8 位转换得到的值存储在 `d` 的低 16 位。可选操作数 `scale-factor` 的类型为 `.b16`，用于存储两个打包的 `.ue8m0` 类型缩放因子。对于向上转换，输入先被转换为目标类型，然后再乘以 `scale-factor`。来自 `a` 高 8 位的 `.s2f6` 输入的缩放因子存储在操作数 `scale-factor` 的高 8 位，来自 `a` 低 8 位的 `.s2f6` 输入的缩放因子存储在操作数 `scale-factor` 的低 8 位。如果未显式指定，`scale-factor` 的值假定为 `0x7f7f`，即两个输入缩放因子均为值 `1`。

可选限定符 `.scaled::n2::ue8m0` 表示该指令使用带有 2 个 `ue8m0` 类型缩放值的打包 `scale-factor`。操作数 `scale-factor` 与限定符 `.scaled::n2::ue8m0` 必须配合使用。

`rbits` 是一个 `.b32` 类型寄存器操作数，用于为 `.rs` 舍入模式提供随机比特。

当转换到 `.f16x2` 时，会从 `rbits` 提供两个 16-bit 值，其中高 16-bit 的 13 个 LSB 用作操作数 `a` 的随机比特（3 个 MSB 为 0），低 16-bit 的 13 个 LSB 用作操作数 `b` 的随机比特（3 个 MSB 为 0）。

当转换到 `.bf16x2` 时，会从 `rbits` 提供两个 16-bit 值，其中高 16-bit 用作操作数 `a` 的随机比特，低 16-bit 用作操作数 `b` 的随机比特。

当转换到 `.e4m3x4`/`.e5m2x4`/`.e2m3x4`/`.e3m2x4` 时，会从 `rbits` 提供两个 16-bit 值，其中低 16-bit 用于操作数 `e`、`f`，高 16-bit 用于操作数 `a`、`b`。

当转换到 `.e2m1x4` 时，会从 `rbits` 提供两个 16-bit 值，其中两个 16-bit 半部的低 8-bit 用于操作数 `e`、`f`，两个 16-bit 半部的高 8-bit 用于操作数 `a`、`b`。

在以下所有情况中，舍入修饰符是必需的:

- float-to-float 转换，且目标类型小于源类型

- 所有 float-to-int 转换

- 所有 int-to-float 转换

- 所有涉及 `.f16x2`、`.e4m3x2, .e5m2x2,`、`.bf16x2`、`.tf32`、`.e2m1x2`、`.e2m3x2`、`.e3m2x2`、`.e4m3x4`、`.e5m2x4`、`.e2m1x4`、`.e2m3x4`、`.e3m2x4`、`.s2f6x2` 和 `.ue8m0x2` 指令类型的转换。

`.satfinite` 修饰符仅支持涉及以下类型的转换:

- 目标类型为 `.e4m3x2`、`.e5m2x2`、`.e2m1x2`、`.e2m3x2`、`.e3m2x2`、`.e4m3x4`、`.e5m2x4`、`.e2m1x4`、`.e2m3x4`、`.e3m2x4`、`.s2f6x2`。此类转换必须使用 `.satfinite` 修饰符。

- 目标类型为 `.f16`、`.bf16`、`.f16x2`、`.bf16x2`、`.tf32`、`.ue8m0x2`。

语义

```
if (/* inst type is .f16x2 or .bf16x2 */) {
    d[31:16] = convert(a);
    d[15:0]  = convert(b);
} else if (/* inst destination type is .e5m2x2 or .e4m3x2 or .ue8m0x2 */) {
    if (/* inst source type is .f32 */) {
        d[15:8] = convert(a);
        d[7:0]  = convert(b);
    } else {
        d[15:8] = convert(a[31:16]);
        d[7:0]  = convert(a[15:0]);
    }
} else if (/* inst destination type is .s2f6x2 */) {
    if (/* inst source type is .f32 */) {
        d[15:8] = convert(a / scale-factor[15:8]);
        d[7:0]  = convert(b / scale-factor[7:0]);
    } else {
        d[15:8] = convert(a[15:8] / scale-factor[15:8]);
        d[7:0]  = convert(a[7:0] / scale-factor[7:0]);
    }
} else if (/* inst source type is .s2f6x2 */) {
        d[31:16] = convert(a[15:8]) * scale-factor[15:8];
        d[15:0]  = convert(a[7:0]) * scale-factor[7:0];
} else if (/* inst destination type is .e2m1x2 */) {
    if (/* inst source type is .f32 */) {
        d[7:4] = convert(a);
        d[3:0] = convert(b);
    } else {
        d[7:4] = convert(a[31:16]);
        d[3:0]  = convert(a[15:0]);
    }
} else if (/* inst destination type is .e2m3x2 or .e3m2x2 */) {
    if (/* inst source type is .f32 */) {
        d[15:14] = 0;
        d[13:8] = convert(a);
        d[7:6] = 0;
        d[5:0] = convert(b);
    } else {
        d[15:14] = 0;
        d[13:8] = convert(a[31:16]);
        d[7:6] = 0;
        d[5:0] = convert(a[15:0]);
    }
} else if (/* inst destination type is .e2m1x4 */) {
    d[15:12] = convert(a);
    d[11:8] = convert(b);
    d[7:4] = convert(e);
    d[3:0] = convert(f);
} else if (/* inst destination type is .e4m3x4 or .e5m2x4 */) {
    d[31:24] = convert(a);
    d[23:16] = convert(b);
    d[15:8] = convert(e);
    d[7:0] = convert(f);
} else if (/* inst destination type is .e2m3x4 or .e3m2x4 */) {
    d[31:30] = 0;
    d[29:24] = convert(a);
    d[23:22] = 0;
    d[21:16] = convert(b);
    d[15:14] = 0;
    d[13:8] = convert(e);
    d[7:6] = 0;
    d[5:0] = convert(f);
} else {
    d = convert(a);
}
```

// Random bits `rbits` semantics for `.rs` rounding:

1. Destination type `.f16`: Refer [Figure 38](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-f16) for random bits layout details.

    ![_images/cvt-rs-rbits-layout-f16.png](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/cvt-rs-rbits-layout-f16.png)

    Figure 38 Random bits layout for `.rs` rounding with `.f16` destination type[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-f16 "Permalink to this image")

2. Destination type `.bf16`: Refer [Figure 39](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-bf16) for random bits layout details.

    ![_images/cvt-rs-rbits-layout-bf16.png](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/cvt-rs-rbits-layout-bf16.png)

    Figure 39 Random bits layout for `.rs` rounding with `.bf16` destination type[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-bf16 "Permalink to this image")

3. Destination type `.e2m1x4`: Refer [Figure 40](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-f4) for random bits layout details.

    ![_images/cvt-rs-rbits-layout-f4.png](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/cvt-rs-rbits-layout-f4.png)

    Figure 40 Random bits layout for `.rs` rounding with `.e2m1x4` destination type[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-f4 "Permalink to this image")

4. Destination type `.e5m2x4`, `.e4m3x4`, `.e3m2x4`, `.e2m3x4`: Refer [Figure 41](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-f8-f6) for random bits layout details.

    ![_images/cvt-rs-rbits-layout-f8-f6.png](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/cvt-rs-rbits-layout-f8-f6.png)

    Figure 41 Random bits layout for `.rs` rounding with `.e5m2x4`/`.e4m3x4`/`.e3m2x4`/`.e2m3x4` destination type[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cvt-rs-rbits-layout-f8-f6 "Permalink to this image")

## 整数说明

浮点到整数转换需要整数舍入，同尺寸的 float-to-float 转换如果结果被舍入为整数，也需要整数舍入。除这些情况外，整数舍入都是非法的。

整数舍入修饰符:

`.rni`

舍入到最近整数，若源值等距于两个整数则取偶数

`.rzi`

向零方向舍入到最近整数

`.rmi`

向负无穷方向舍入到最近整数

`.rpi`

向正无穷方向舍入到最近整数

在 float-to-int 转换中，根据转换类型，`NaN` 输入会得到如下值:

1. 当源不是 `.f64` 且目标不是 `.s64`、`.u64` 时结果为 0。

2. 否则为 1 << (BitWidth(dst) - 1)，对应无符号类型的 (`MAXINT` >> 1) + 1 或有符号类型的 `MININT`。

次正规数:

`sm_20+`

默认情况下支持次正规数。

对于 `cvt.ftz.dtype.f32` 的 float-to-integer 转换，以及带整数舍入的 `cvt.ftz.f32.f32` float-to-float 转换，次正规输入会被冲洗为保留符号的零。修饰符 `.ftz` 只能在 `.dtype` 或 `.atype` 为 `.f32` 时指定，且只适用于单精度 (`.f32`) 输入和结果。

`sm_1x`

对于 `cvt.ftz.dtype.f32` 的 float-to-integer 转换，以及带整数舍入的 `cvt.ftz.f32.f32` float-to-float 转换，次正规输入会被冲洗为保留符号的零。在这些情况下可选的 `.ftz` 修饰符可用于明确表达。

**Note:** 在 PTX ISA 版本 1.4 及更早版本中，如果目标类型大小为 64-bit，`cvt` 指令不会将单精度次正规输入或结果冲洗为零。编译器将为旧版 PTX 代码保留此行为。

饱和修饰符:

`.sat`

对于整数目标类型，`.sat` 将结果限制在该操作尺寸的 `MININT..MAXINT` 范围内。注意饱和同时适用于有符号和无符号整数类型。

`.sat` 修饰符只允许在目标类型的取值范围不是源类型取值范围的超集时使用; 即在无法发生饱和的情况下，`.sat` 修饰符是非法的。

对于 float-to-integer 转换，结果默认被夹紧到目标范围; 即 `.sat` 是冗余的。

## 浮点说明

当 float-to-float 转换会导致精度损失时，以及 int-to-float 转换时，需要进行浮点舍入。在其他情况下，浮点舍入是非法的。

浮点舍入修饰符:

`.rn`

舍入到最近值，若等距则取偶数

`.rna`

舍入到最近值，若等距则向远离零的方向

`.rz`

向零方向舍入

`.rm`

向负无穷方向舍入

`.rp`

向正无穷方向舍入

`.rs`

通过使用提供的随机比特实现随机舍入。操作结果基于将提供的随机比特 (`rbits`) 与输入尾数中被截断（丢弃）的比特做整数加法的进位结果，决定向零方向还是远离零方向舍入。

浮点值可以使用整数舍入修饰符（见整数说明）来舍入为整数值。操作数必须具有相同位宽。结果为以浮点格式存储的整数值。

次正规数:

`sm_20+`

默认情况下支持次正规数。可指定 `.ftz` 修饰符将单精度次正规输入与结果冲洗为保留符号的零。`.ftz` 修饰符只能在 `.dtype` 或 `.atype` 为 `.f32` 时指定，且仅适用于单精度 (`.f32`) 输入和结果。

`sm_1x`

单精度次正规输入与结果会被冲洗为保留符号的零。在这些情况下可选的 `.ftz` 修饰符可用于明确表达。

**Note:** 在 PTX ISA 版本 1.4 及更早版本中，若源或目标类型为 `.f64`，`cvt` 指令不会将单精度次正规输入或结果冲洗为零。编译器将为旧版 PTX 代码保留此行为。具体而言，如果 PTX ISA 版本为 1.4 或更早，则仅对 `cvt.f32.f16`、`cvt.f16.f32` 与 `cvt.f32.f32` 指令将单精度次正规输入和结果冲洗为保留符号的零。

饱和修饰符:

`.sat`:

对于浮点目标类型，`.sat` 将结果限制在范围 [0.0, 1.0] 内。`NaN` 结果被冲洗为正零。适用于 `.f16`、`.f32` 和 `.f64` 类型。

`.relu`:

对于 `.f16`、`.f16x2`、`.bf16`、`.bf16x2`、`.e4m3x2`、`.e5m2x2`、`.e2m1x2`、`.e2m3x2`、`.e3m2x2`、`.e4m3x4`、`.e5m2x4`、`.e2m1x4`、`.e2m3x4`、`.e3m2x4`、`.s2f6x2` 和 `.tf32` 目标类型，`.relu` 会将负值结果截断为 0。`NaN` 结果会被转换为规范 `NaN`。

`.satfinite`:

对于 `.f16`、`.f16x2`、`.bf16`、`.bf16x2`、`.e4m3x2`、`.e5m2x2`、`.ue8m0x2`、`.e4m3x4`、`.e5m2x4` 和 `.tf32` 目标格式，如果输入值为 `NaN`，则结果为指定目标格式中的 `NaN`。对于 `.e2m1x2`、`.e2m3x2`、`.e3m2x2`、`.e2m1x4`、`.e2m3x4`、`.e3m2x4`、`.s2f6x2` 目标格式，`NaN` 结果会被转换为正的 _MAX_NORM_。如果输入绝对值（忽略符号）大于指定目标格式的 _MAX_NORM_，则结果为目标格式中保留符号的 _MAX_NORM_，并且对不支持目标符号的 `.ue8m0x2`，结果为正的 _MAX_NORM_。

备注

可使用比指定类型更宽的源寄存器，但当源操作数为 `.bf16` 或 `.bf16x2` 格式时除外。用于转换的仅是与指令类型宽度对应的低 `n` 位。关于这些宽松类型检查规则，请参见 [Operand Size Exceeding Instruction-Type Size](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operand-size-exceeding-instruction-type-size)。

可使用比指定类型更宽的目标寄存器，但当目标操作数为 `.bf16`、`.bf16x2` 或 `.tf32` 格式时除外。对于有符号整数类型，转换结果会进行符号扩展至目标寄存器宽度; 对于无符号、位大小和浮点类型，转换结果会进行零扩展至目标寄存器宽度。关于这些宽松类型检查规则，请参见 [Operand Size Exceeding Instruction-Type Size](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operand-size-exceeding-instruction-type-size)。

对于 `cvt.f32.bf16`，`NaN` 输入会产生未指定的 `NaN`。

## PTX ISA 说明

PTX ISA 版本 1.0 引入。

`.relu` 修饰符与 {`.f16x2`, `.bf16`, `.bf16x2`, `.tf32`} 目标格式在 PTX ISA 版本 7.0 引入。

`cvt.f32.bf16` 在 PTX ISA 版本 7.1 引入。

`cvt.bf16.{u8/s8/u16/s16/u32/s32/u64/s64/f16/f64/bf16}`, `cvt.{u8/s8/u16/s16/u32/s32/u64/s64/f16/f64}.bf16`, 和 `cvt.tf32.f32.{relu}.{rn/rz}` 在 PTX ISA 版本 7.8 引入。

`cvt.f32.bf16` 的 `.ftz` 限定符在 PTX ISA 版本 7.8 引入。

面向 `sm_90` 或更高版本的 `.e4m3x2`/`.e5m2x2` `cvt` 在 PTX ISA 版本 7.8 引入。

面向 `sm_90` 或更高版本的 `cvt.satfinite.{e4m3x2, e5m2x2}.{f32, f16x2}` 在 PTX ISA 版本 7.8 引入。

面向 `sm_89` 的 `.e4m3x2`/`.e5m2x2` `cvt` 在 PTX ISA 版本 8.1 引入。

面向 `sm_89` 的 `cvt.satfinite.{e4m3x2, e5m2x2}.{f32, f16x2}` 在 PTX ISA 版本 8.1 引入。

`cvt.satfinite.{f16, bf16, f16x2, bf16x2, tf32}.f32` 在 PTX ISA 版本 8.1 引入。

`cvt.{rn/rz}.satfinite.tf32.f32` 在 PTX ISA 版本 8.6 引入。

`cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32` 在 PTX ISA 版本 8.6 引入。

`cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2}` 在 PTX ISA 版本 8.6 引入。

`cvt.{rp/rz}{.satfinite}{.relu}.ue8m0x2.bf16x2` 在 PTX ISA 版本 8.6 引入。

`cvt.{rz/rp}.satfinite.ue8m0x2.f32` 在 PTX ISA 版本 8.6 引入。

`cvt.rn.bf16x2.ue8m0x2` 在 PTX ISA 版本 8.6 引入。

`.rs` 舍入模式在 PTX ISA 版本 8.7 引入。

`cvt.rs{.e2m1x4/.e4m3x4/.e5m2x4/.e3m2x4/.e2m3x4}.f32` 在 PTX ISA 版本 8.7 引入。

`cvt.rn.satfinite{.relu}{.e5m2x2/.e4m3x2}{.bf16x2}` 在 PTX ISA 版本 9.1 引入。

`cvt.rn.satfinite{.relu}{.e2m3x2/.e3m2x2/.e2m1x2}{.f16x2/.bf16x2}` 在 PTX ISA 版本 9.1 引入。

`.s2f6x2` 指令类型的 `cvt` 在 PTX ISA 版本 9.1 引入。

## Target ISA 说明

将 `cvt` 转换到或从 `.f64` 需要 `sm_13` 或更高版本。

`.relu` 修饰符与 {`.f16x2`, `.bf16`, `.bf16x2`, `.tf32`} 目标格式需要 `sm_80` 或更高版本。

`cvt.f32.bf16` 需要 `sm_80` 或更高版本。

`cvt.bf16.{u8/s8/u16/s16/u32/s32/u64/s64/f16/f64/bf16}`, `cvt.{u8/s8/u16/s16/u32/s32/u64/s64/f16/f64}.bf16`, 和 `cvt.tf32.f32.{relu}.{rn/rz}` 需要 `sm_90` 或更高版本。

`cvt.f32.bf16` 的 `.ftz` 限定符需要 `sm_90` 或更高版本。

带 `.e4m3x2`/`.e5m2x2` 的 `cvt` 需要 `sm89` 或更高版本。

`cvt.satfinite.{e4m3x2, e5m2x2}.{f32, f16x2}` 需要 `sm_89` 或更高版本。

`cvt.{rn/rz}.satfinite.tf32.f32` 需要 `sm_100` 或更高版本。

`cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32` 支持以下架构:

- `sm_100a`

- `sm_101a` (从 PTX ISA 版本 9.0 起更名为 `sm_110a`)

- `sm_120a`

- 并从 PTX ISA 版本 8.8 起支持以下家族特定架构:

    - `sm_100f` 或更高，同一家族内

    - `sm_101f` 或更高，同一家族内 (从 PTX ISA 版本 9.0 起更名为 `sm_110f`)

    - `sm_120f` 或更高，同一家族内

- `sm_110f` 或更高，同一家族内

`cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2}` 支持以下架构:

- `sm_100a`

- `sm_101a` (从 PTX ISA 版本 9.0 起更名为 `sm_110a`)

- `sm_120a`

- 并从 PTX ISA 版本 8.8 起支持以下家族特定架构:

    - `sm_100f` 或更高，同一家族内

    - `sm_101f` 或更高，同一家族内 (从 PTX ISA 版本 9.0 起更名为 `sm_110f`)

    - `sm_120f` 或更高，同一家族内

- `sm_110f` 或更高，同一家族内

`cvt.{rz/rp}{.satfinite}{.relu}.ue8m0x2.bf16x2` 支持以下架构:

- `sm_100a`

- `sm_101a` (从 PTX ISA 版本 9.0 起更名为 `sm_110a`)

- `sm_120a`

- 并从 PTX ISA 版本 8.8 起支持以下家族特定架构:

    - `sm_100f` 或更高，同一家族内

    - `sm_101f` 或更高，同一家族内 (从 PTX ISA 版本 9.0 起更名为 `sm_110f`)

    - `sm_120f` 或更高，同一家族内

- `sm_110f` 或更高，同一家族内

`cvt.{rz/rp}.satfinite.ue8m0x2.f32` 支持以下架构:

- `sm_100a`

- `sm_101a` (从 PTX ISA 版本 9.0 起更名为 `sm_110a`)

- `sm_120a`

- 并从 PTX ISA 版本 8.8 起支持以下家族特定架构:

    - `sm_100f` 或更高，同一家族内

    - `sm_101f` 或更高，同一家族内 (从 PTX ISA 版本 9.0 起更名为 `sm_110f`)

    - `sm_120f` 或更高，同一家族内

- `sm_110f` 或更高，同一家族内

`cvt.rn.bf16x2.ue8m0x2` 支持以下架构:

- `sm_100a`

- `sm_101a` (从 PTX ISA 版本 9.0 起更名为 `sm_110a`)

- `sm_120a`

- 并从 PTX ISA 版本 8.8 起支持以下家族特定架构:

    - `sm_100f` 或更高，同一家族内

    - `sm_101f` 或更高，同一家族内 (从 PTX ISA 版本 9.0 起更名为 `sm_110f`)

    - `sm_120f` 或更高，同一家族内

- `sm_110f` 或更高，同一家族内

`.rs` 舍入模式支持以下架构:

- `sm_100a`

- `sm_103a`

`cvt.rs{.e2m1x4/.e4m3x4/.e5m2x4/.e3m2x4/.e2m3x4}.f32` 支持以下架构:

- `sm_100a`

- `sm_103a`

`cvt.rn.satfinite{.relu}{.e5m2x2/.e4m3x2}{.bf16x2}` 支持以下家族特定架构:

- `sm_100f` 或更高，同一家族内

- `sm_110f` 或更高，同一家族内

- `sm_120f` 或更高，同一家族内

`cvt.rn.satfinite{.relu}{.e2m3x2/.e3m2x2/.e2m1x2}{.f16x2/.bf16x2}` 支持以下家族特定架构:

- `sm_100f` 或更高，同一家族内

- `sm_110f` 或更高，同一家族内

- `sm_120f` 或更高，同一家族内

`.s2f6x2` 指令类型的 `cvt` 支持以下架构:

- `sm_100a`

- `sm_103a`

- `sm_110a`

- `sm_120a`

- `sm_121a`

## 示例

```
cvt.f32.s32 f,i;
cvt.s32.f64 j,r;     // float-to-int saturates by default
cvt.rni.f32.f32 x,y; // round to nearest int, result is fp
cvt.f32.f32 x,y;     // note .ftz behavior for sm_1x targets
cvt.rn.relu.f16.f32      b, f;        // result is saturated with .relu saturation mode
cvt.rz.f16x2.f32         b1, f, f1;   // convert two fp32 values to packed fp16 outputs
cvt.rn.relu.satfinite.f16x2.f32    b1, f, f1;   // convert two fp32 values to packed fp16 outputs with .relu saturation on each output
cvt.rn.bf16.f32          b, f;        // convert fp32 to bf16
cvt.rz.relu.satfinite.bf16.f3 2    b, f;        // convert fp32 to bf16 with .relu and .satfinite saturation
cvt.rz.satfinite.bf16x2.f32        b1, f, f1;   // convert two fp32 values to packed bf16 outputs
cvt.rn.relu.bf16x2.f32   b1, f, f1;   // convert two fp32 values to packed bf16 outputs with .relu saturation on each output
cvt.rna.satfinite.tf32.f32         b1, f;       // convert fp32 to tf32 format
cvt.rn.relu.tf32.f32     d, a;        // convert fp32 to tf32 format
cvt.f64.bf16.rp          f, b;        // convert bf16 to f64 format
cvt.bf16.f16.rz          b, f         // convert f16 to bf16 format
cvt.bf16.u64.rz          b, u         // convert u64 to bf16 format
cvt.s8.bf16.rpi          s, b         // convert bf16 to s8 format
cvt.bf16.bf16.rpi        b1, b2       // convert bf16 to corresponding int represented in bf16 format
cvt.rn.satfinite.e4m3x2.f32 d, a, b;  // convert a, b to .e4m3 and pack as .e4m3x2 output
cvt.rn.relu.satfinite.e5m2x2.f16x2 d, a; // unpack a and convert the values to .e5m2 outputs with .relu
                                         // saturation on each output and pack as .e5m2x2
cvt.rn.f16x2.e4m3x2 d, a;             // unpack a, convert two .e4m3 values to packed f16x2 output
cvt.rn.satfinite.tf32.f32 d, a;       // convert fp32 to tf32 format
cvt.rn.relu.f16x2.e2m1x2 d, a;        // unpack a, convert two .e2m1 values to packed f16x2 output
cvt.rn.satfinite.e2m3x2.f32 d, a, b;  // convert a, b to .e2m3 and pack as .e2m3x2 output
cvt.rn.relu.f16x2.e3m2x2 d, a;        // unpack a, convert two .e3m2 values to packed f16x2 output

cvt.rs.f16x2.f32    d, a, b, rbits;  // convert 2 fp32 values to packed fp16 with applying .rs rounding
cvt.rs.satfinite.e2m1x4.f32  d, {a, b, e, f}, rbits; // convert 4 fp32 values to packed 4 e2m1 values with applying .rs rounding

cvt.rn.satfinite.relu.e2m1x2type.f16x2  d, a; // unpack a and covert to two .e2m1 values
cvt.rn.satfinite.e2m3x2type.bf16x2  d, a; // unpack a and covert to two .e2m3 values

// Convert 2 f32 values to s2f6 after applying scale factor for dividing the value
cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32 d, a, b, scale-factor;
```