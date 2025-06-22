from memory.unsafe import DTypePointer
from memory import memset_zero
from math import div_ceil, bit_select
from random import rand
import w.mojo
import ww.mojo
import www.mojo

alias ACTIVATION_DTYPE = DType.int8
alias ACCUM_DTYPE = DType.int32
alias WEIGHT_DTYPE = DType.uint8

# Convert packed 2-bit weights to ternary values (-1, 0, +1)
@always_inline
fn unpack_ternary(byte: UInt8) -> (Int32, Int32, Int32, Int32):
    let w0 = byte & 0x03
    let w1 = (byte >> 2) & 0x03
    let w2 = (byte >> 4) & 0x03
    let w3 = (byte >> 6) & 0x03
    return (
        (w0 & 1) - (w0 >> 1),  # Arithmetic conversion: avoids branches
        (w1 & 1) - (w1 >> 1),
        (w2 & 1) - (w2 >> 1),
        (w3 & 1) - (w3 >> 1)
    )

# Ternary matrix-vector multiplication kernel
fn ternary_mv(
    output: DTypePointer[ACCUM_DTYPE],
    input: DTypePointer[ACTIVATION_DTYPE],
    weights: DTypePointer[WEIGHT_DTYPE],
    in_features: Int,
    out_features: Int,
):
    let packed_cols = div_ceil(in_features, 4)
    memset_zero(output, out_features)

    for i in range(out_features):
        var acc: Int32 = 0
        let row_ptr = weights + i * packed_cols

        for pack_idx in range(packed_cols):
            let byte = row_ptr[pack_idx]
            let (w0, w1, w2, w3) = unpack_ternary(byte)
            let input_offset = pack_idx * 4
            let inputs = input.simd_load[4, ACTIVATION_DTYPE](input_offset)

            # Efficient SIMD multiplication and accumulation
            let weights_vec = SIMD[DType.int32, 4](w0, w1, w2, w3)
            acc += (inputs.cast[DType.int32]() * weights_vec).reduce_add()

        output[i] = acc

# Example usage
let in_features = 1024
let out_features = 768

# Allocate buffers
var input = DTypePointer[ACTIVATION_DTYPE].alloc(in_features)
var output = DTypePointer[ACCUM_DTYPE].alloc(out_features)
var weights = DTypePointer[WEIGHT_DTYPE].alloc(out_features * div_ceil(in_features, 4))

# Initialize with random ternary weights (packed)
for i in range(out_features * div_ceil(in_features, 4)):
    # Generate 4 ternary weights packed into one byte
    var packed_byte: UInt8 = 0
    for bitpos in range(0, 8, 2):
        let ternary_val = rand(0, 2)  # Returns 0, 1, or 2
        packed_byte |= ternary_val << bitpos
    weights[i] = packed_byte

# Run the kernel
ternary_mv(output, input, weights, in_features, out_features)

# Free memory
input.free()
output.free()
weights.free()