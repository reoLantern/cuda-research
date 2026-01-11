#### 9.7.1.3. [Integer Arithmetic Instructions: `mul`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul "Permalink to this headline")

`mul`

Multiply two values.

Syntax

mul.mode.type  d, a, b;

.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };

Description

Compute the product of two values.

Semantics

t = a * b;
n = bitwidth of type;
d = t;            // for .wide
d = t<2n-1..n>;   // for .hi variant
d = t<n-1..0>;    // for .lo variant

Notes

The type of the operation represents the types of the `a` and `b` operands. If `.hi` or `.lo` is specified, then `d` is the same size as `a` and `b`, and either the upper or lower half of the result is written to the destination register. If `.wide` is specified, then `d` is twice as wide as `a` and `b` to receive the full result of the multiplication.

The `.wide` suffix is supported only for 16- and 32-bit integer types.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

mul.wide.s16 fa,fxs,fys;   // 16*16 bits yields 32 bits
mul.lo.s16 fa,fxs,fys;     // 16*16 bits, save only the low 16 bits
mul.wide.s32 z,x,y;        // 32*32 bits, creates 64 bit result

#### 9.7.1.4. [Integer Arithmetic Instructions: `mad`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad "Permalink to this headline")

`mad`

Multiply two values, optionally extract the high or low half of the intermediate result, and add a third value.

Syntax

mad.mode.type  d, a, b, c;
mad.hi.sat.s32 d, a, b, c;

.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };

Description

Multiplies two values, optionally extracts the high or low half of the intermediate result, and adds a third value. Writes the result into a destination register.

Semantics

t = a * b;
n = bitwidth of type;
d = t + c;           // for .wide
d = t<2n-1..n> + c;  // for .hi variant
d = t<n-1..0> + c;   // for .lo variant

Notes

The type of the operation represents the types of the `a` and `b` operands. If .hi or .lo is specified, then `d` and `c` are the same size as `a` and `b`, and either the upper or lower half of the result is written to the destination register. If `.wide` is specified, then `d` and `c` are twice as wide as `a` and `b` to receive the result of the multiplication.

The `.wide` suffix is supported only for 16-bit and 32-bit integer types.

Saturation modifier:

`.sat`

limits result to `MININT..MAXINT` (no overflow) for the size of the operation.

Applies only to `.s32` type in `.hi` mode.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

@p  mad.lo.s32 d,a,b,c;
    mad.lo.s32 r,p,q,r;

#### 9.7.1.7. [Integer Arithmetic Instructions: `sad`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad "Permalink to this headline")

`sad`

Sum of absolute differences.

Syntax

sad.type  d, a, b, c;

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };

Description

Adds the absolute value of `a-b` to `c` and writes the resulting value into `d`.

Semantics

d = c + ((a<b) ? b-a : a-b);

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

sad.s32  d,a,b,c;
sad.u32  d,a,b,d;  // running sum

#### 9.7.1.8. [Integer Arithmetic Instructions: `div`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div "Permalink to this headline")

`div`

Divide one value by another.

Syntax

div.type  d, a, b;

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };

Description

Divides `a` by `b`, stores result in `d`.

Semantics

d = a / b;

Notes

Division by zero yields an unspecified, machine-specific value.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

div.s32  b,n,i;

#### 9.7.1.9. [Integer Arithmetic Instructions: `rem`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem "Permalink to this headline")

`rem`

The remainder of integer division.

Syntax

rem.type  d, a, b;

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };

Description

Divides `a` by `b`, store the remainder in `d`.

Semantics

d = a % b;

Notes

The behavior for negative numbers is machine-dependent and depends on whether divide rounds towards zero or negative infinity.

Division by zero yields an unspecified, machine-specific value.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

rem.s32  x,x,8;    // x = x%8;

#### 9.7.1.10. [Integer Arithmetic Instructions: `abs`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs "Permalink to this headline")

`abs`

Absolute value.

Syntax

abs.type  d, a;

.type = { .s16, .s32, .s64 };

Description

Take the absolute value of `a` and store it in `d`.

Semantics

d = |a|;

Notes

Only for signed integers.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

abs.s32  r0,a;

#### 9.7.1.11. [Integer Arithmetic Instructions: `neg`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg "Permalink to this headline")

`neg`

Arithmetic negate.

Syntax

neg.type  d, a;

.type = { .s16, .s32, .s64 };

Description

Negate the sign of **a** and store the result in **d**.

Semantics

d = -a;

Notes

Only for signed integers.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

neg.s32  r0,a;

#### 9.7.1.15. [Integer Arithmetic Instructions: `clz`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz "Permalink to this headline")

`clz`

Count leading zeros.

Syntax

clz.type  d, a;

.type = { .b32, .b64 };

Description

Count the number of leading zeros in `a` starting with the most-significant bit and place the result in 32-bit destination register `d`. Operand `a` has the instruction type, and destination `d` has type `.u32`. For `.b32` type, the number of leading zeros is between 0 and 32, inclusively. For `.b64` type, the number of leading zeros is between 0 and 64, inclusively.

Semantics

.u32  d = 0;
if (.type == .b32)   { max = 32; mask = 0x80000000; }
else                 { max = 64; mask = 0x8000000000000000; }

while (d < max && (a&mask == 0) ) {
    d++;
    a = a << 1;
}

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`clz` requires `sm_20` or higher.

Examples

clz.b32  d, a;
clz.b64  cnt, X;  // cnt is .u32

#### 9.7.1.16. [Integer Arithmetic Instructions: `bfind`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind "Permalink to this headline")

`bfind`

Find most significant non-sign bit.

Syntax

bfind.type           d, a;
bfind.shiftamt.type  d, a;

.type = { .u32, .u64,
          .s32, .s64 };

Description

Find the bit position of the most significant non-sign bit in `a` and place the result in `d`. Operand `a` has the instruction type, and destination `d` has type `.u32`. For unsigned integers, `bfind` returns the bit position of the most significant `1`. For signed integers, `bfind` returns the bit position of the most significant `0` for negative inputs and the most significant `1` for non-negative inputs.

If `.shiftamt` is specified, `bfind` returns the shift amount needed to left-shift the found bit into the most-significant bit position.

`bfind` returns `0xffffffff` if no non-sign bit is found.

Semantics

msb = (.type==.u32 || .type==.s32) ? 31 : 63;
// negate negative signed inputs
if ( (.type==.s32 || .type==.s64) && (a & (1<<msb)) ) {
    a = ~a;
}
.u32  d = 0xffffffff;
for (.s32 i=msb; i>=0; i--) {
    if (a & (1<<i))  { d = i; break; }
}
if (.shiftamt && d != 0xffffffff)  { d = msb - d; }

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`bfind` requires `sm_20` or higher.

Examples

bfind.u32  d, a;
bfind.shiftamt.s64  cnt, X;  // cnt is .u32

#### 9.7.1.17. [Integer Arithmetic Instructions: `fns`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns "Permalink to this headline")

`fns`

Find the n-th set bit

Syntax

fns.b32 d, mask, base, offset;

Description

Given a 32-bit value `mask` and an integer value `base` (between 0 and 31), find the n-th (given by offset) set bit in `mask` from the `base` bit, and store the bit position in `d`. If not found, store 0xffffffff in `d`.

Operand `mask` has a 32-bit type. Operand `base` has `.b32`, `.u32` or `.s32` type. Operand offset has `.s32` type. Destination `d` has type `.b32.`

Operand `base` must be <= 31, otherwise behavior is undefined.

Semantics

d = 0xffffffff;
if (offset == 0) {
    if (mask[base] == 1) {
        d = base;
    }
} else {
    pos = base;
    count = |offset| - 1;
    inc = (offset > 0) ? 1 : -1;

    while ((pos >= 0) && (pos < 32)) {
        if (mask[pos] == 1) {
            if (count == 0) {
              d = pos;
              break;
           } else {
               count = count - 1;
           }
        }
        pos = pos + inc;
    }
}

PTX ISA Notes

Introduced in PTX ISA version 6.0.

Target ISA Notes

`fns` requires `sm_30` or higher.

Examples

fns.b32 d, 0xaaaaaaaa, 3, 1;   // d = 3
fns.b32 d, 0xaaaaaaaa, 3, -1;  // d = 3
fns.b32 d, 0xaaaaaaaa, 2, 1;   // d = 3
fns.b32 d, 0xaaaaaaaa, 2, -1;  // d = 1

#### 9.7.1.18. [Integer Arithmetic Instructions: `brev`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev "Permalink to this headline")

`brev`

Bit reverse.

Syntax

brev.type  d, a;

.type = { .b32, .b64 };

Description

Perform bitwise reversal of input.

Semantics

msb = (.type==.b32) ? 31 : 63;

for (i=0; i<=msb; i++) {
    d[i] = a[msb-i];
}

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`brev` requires `sm_20` or higher.

Examples

brev.b32  d, a;

#### 9.7.1.19. [Integer Arithmetic Instructions: `bfe`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe "Permalink to this headline")

`bfe`

Bit Field Extract.

Syntax

bfe.type  d, a, b, c;

.type = { .u32, .u64,
          .s32, .s64 };

Description

Extract bit field from `a` and place the zero or sign-extended result in `d`. Source `b` gives the bit field starting bit position, and source `c` gives the bit field length in bits.

Operands `a` and `d` have the same type as the instruction type. Operands `b` and `c` are type `.u32`, but are restricted to the 8-bit value range `0..255`.

The sign bit of the extracted field is defined as:

`.u32`, `.u64`:

zero

`.s32`, `.s64`:

`msb` of input a if the extracted field extends beyond the `msb` of a `msb` of extracted field, otherwise

If the bit field length is zero, the result is zero.

The destination `d` is padded with the sign bit of the extracted field. If the start position is beyond the `msb` of the input, the destination `d` is filled with the replicated sign bit of the extracted field.

Semantics

msb = (.type==.u32 || .type==.s32) ? 31 : 63;
pos = b & 0xff;  // pos restricted to 0..255 range
len = c & 0xff;  // len restricted to 0..255 range

if (.type==.u32 || .type==.u64 || len==0)
    sbit = 0;
else
    sbit = a[min(pos+len-1,msb)];

d = 0;
for (i=0; i<=msb; i++) {
    d[i] = (i<len && pos+i<=msb) ? a[pos+i] : sbit;
}

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`bfe` requires `sm_20` or higher.

Examples

bfe.b32  d,a,start,len;

#### 9.7.1.20. [Integer Arithmetic Instructions: `bfi`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi "Permalink to this headline")

`bfi`

Bit Field Insert.

Syntax

bfi.type  f, a, b, c, d;

.type = { .b32, .b64 };

Description

Align and insert a bit field from `a` into `b`, and place the result in `f`. Source `c` gives the starting bit position for the insertion, and source `d` gives the bit field length in bits.

Operands `a`, `b`, and `f` have the same type as the instruction type. Operands `c` and `d` are type `.u32`, but are restricted to the 8-bit value range `0..255`.

If the bit field length is zero, the result is `b`.

If the start position is beyond the msb of the input, the result is `b`.

Semantics

msb = (.type==.b32) ? 31 : 63;
pos = c & 0xff;  // pos restricted to 0..255 range
len = d & 0xff;  // len restricted to 0..255 range

f = b;
for (i=0; i<len && pos+i<=msb; i++) {
    f[pos+i] = a[i];
}

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`bfi` requires `sm_20` or higher.

Examples

bfi.b32  d,a,b,start,len;

#### 9.7.1.21. [Integer Arithmetic Instructions: `szext`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext "Permalink to this headline")

`szext`

Sign-extend or Zero-extend.

Syntax

szext.mode.type  d, a, b;

.mode = { .clamp, .wrap };
.type = { .u32, .s32 };

Description

Sign-extends or zero-extends an N-bit value from operand `a` where N is specified in operand `b`. The resulting value is stored in the destination operand `d`.

For the `.s32` instruction type, the value in `a` is treated as an N-bit signed value and the most significant bit of this N-bit value is replicated up to bit 31. For the `.u32` instruction type, the value in `a` is treated as an N-bit unsigned number and is zero-extended to 32 bits. Operand `b` is an unsigned 32-bit value.

If the value of N is 0, then the result of `szext` is 0. If the value of N is 32 or higher, then the result of `szext` depends upon the value of the `.mode` qualifier as follows:

- If `.mode` is `.clamp`, then the result is the same as the source operand `a`.
    
- If `.mode` is `.wrap`, then the result is computed using the wrapped value of N.
    

Semantics

b1        = b & 0x1f;
too_large = (b >= 32 && .mode == .clamp) ? true : false;
mask      = too_large ? 0 : (~0) << b1;
sign_pos  = (b1 - 1) & 0x1f;

if (b1 == 0 || too_large || .type != .s32) {
    sign_bit = false;
} else {
    sign_bit = (a >> sign_pos) & 1;
}
d = (a & ~mask) | (sign_bit ? mask | 0);

PTX ISA Notes

Introduced in PTX ISA version 7.6.

Target ISA Notes

`szext` requires `sm_70` or higher.

Examples

szext.clamp.s32 rd, ra, rb;
szext.wrap.u32  rd, 0xffffffff, 0; // Result is 0.

#### 9.7.1.22. [Integer Arithmetic Instructions: `bmsk`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bmsk)[](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bmsk "Permalink to this headline")

`bmsk`

Bit Field Mask.

Syntax

bmsk.mode.b32  d, a, b;

.mode = { .clamp, .wrap };

Description

Generates a 32-bit mask starting from the bit position specified in operand `a`, and of the width specified in operand `b`. The generated bitmask is stored in the destination operand `d`.

The resulting bitmask is 0 in the following cases:

- When the value of `a` is 32 or higher and `.mode` is `.clamp`.
    
- When either the specified value of `b` or the wrapped value of `b` (when `.mode` is specified as `.wrap`) is 0.
    

Semantics

a1    = a & 0x1f;
mask0 = (~0) << a1;
b1    = b & 0x1f;
sum   = a1 + b1;
mask1 = (~0) << sum;

sum-overflow          = sum >= 32 ? true : false;
bit-position-overflow = false;
bit-width-overflow    = false;

if (.mode == .clamp) {
    if (a >= 32) {
        bit-position-overflow = true;
        mask0 = 0;
    }
    if (b >= 32) {
        bit-width-overflow = true;
    }
}

if (sum-overflow || bit-position-overflow || bit-width-overflow) {
    mask1 = 0;
} else if (b1 == 0) {
    mask1 = ~0;
}
d = mask0 & ~mask1;

Notes

The bitmask width specified by operand `b` is limited to range `0..32` in `.clamp` mode and to range `0..31` in `.wrap` mode.

PTX ISA Notes

Introduced in PTX ISA version 7.6.

Target ISA Notes

`bmsk` requires `sm_70` or higher.

Examples

bmsk.clamp.b32  rd, ra, rb;
bmsk.wrap.b32   rd, 1, 2; // Creates a bitmask of 0x00000006.