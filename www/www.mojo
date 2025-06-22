fn ternary_mv(
    output: DTypePointer[ACCUM_DTYPE],
    input: DTypePointer[ACTIVATION_DTYPE],
    weights: DTypePointer[WEIGHT_DTYPE],
    in_features: Int,
    out_features: Int,
):
    let packed_cols = div_ceil(in_features, 4)
    let full_blocks = in_features // 4
    let tail_elements = in_features % 4
    
    memset_zero(output, out_features)

    for i in range(out_features):
        var acc: Int32 = 0
        let row_ptr = weights + i * packed_cols

        # Process complete 4-weight blocks
        for pack_idx in range(full_blocks):
            let byte = row_ptr[pack_idx]
            let (w0, w1, w2, w3) = unpack_ternary(byte)
            let input_offset = pack_idx * 4
            let inputs = input.simd_load[4, ACTIVATION_DTYPE](input_offset)

            let weights_vec = SIMD[DType.int32, 4](w0, w1, w2, w3)
            acc += (inputs.cast[DType.int32]() * weights_vec).reduce_add()

        # Process remaining elements (tail handling)
        if tail_elements > 0:
            let byte = row_ptr[full_blocks]
            let (w0, w1, w2, w3) = unpack_ternary(byte)
            let input_offset = full_blocks * 4
            
            # Manually handle remaining 1-3 elements
            if tail_elements > 0:
                acc += input[input_offset] * w0
            if tail_elements > 1:
                acc += input[input_offset + 1] * w1
            if tail_elements > 2:
                acc += input[input_offset + 2] * w2

        output[i] = acc