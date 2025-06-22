@always_inline
fn unpack_ternary(byte: UInt8) -> (Int32, Int32, Int32, Int32):
    # Extract 2-bit weights from packed byte
    let w0 = byte & 0x03
    let w1 = (byte >> 2) & 0x03
    let w2 = (byte >> 4) & 0x03
    let w3 = (byte >> 6) & 0x03

    # Convert 2-bit patterns to ternary values (-1, 0, +1)
    # 00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0
    let v0 = (w0 & 1) - (w0 >> 1)
    let v1 = (w1 & 1) - (w1 >> 1)
    let v2 = (w2 & 1) - (w2 >> 1)
    let v3 = (w3 & 1) - (w3 >> 1)

    return (v0, v1, v2, v3)