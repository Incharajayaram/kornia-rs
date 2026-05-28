//! CPU-side precompute of bilinear weight tables.
//!
//! For each output coordinate, we store `(src_idx, weight_x256)` where
//! `weight_x256 ∈ [0, 256]` is the fractional weight times 256. The output
//! sample is then computed by the kernel as
//! `((256 - w) * src[idx] + w * src[idx + 1] + 128) >> 8`.
//!
//! Ported verbatim from `crates/kornia-cubecl/src/resize/weights.rs` so that
//! both prototypes produce identical weight tables for the same `(src_len, dst_len)`.
//! Bit-exact agreement with `fast_image_resize` depends on this rounding policy.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisWeight {
    pub src_idx: u32,
    pub weight_x256: u16,
}

/// Compute axis weights for a 1D resize from `src_len` to `dst_len`.
///
/// Uses pixel-centered sampling: output pixel `i` samples at source coordinate
/// `(i + 0.5) * src_len / dst_len - 0.5`, clamped to `[0, src_len - 1]`. This
/// matches `fast_image_resize`'s default sampling convention so cross-impl
/// outputs can be compared within ±1 LSB.
pub fn compute_axis_weights(src_len: u32, dst_len: u32) -> Vec<AxisWeight> {
    assert!(src_len >= 2, "src_len must be at least 2 for bilinear");
    assert!(dst_len >= 1, "dst_len must be at least 1");
    let scale = src_len as f64 / dst_len as f64;
    (0..dst_len)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale - 0.5;
            let center = center.clamp(0.0, (src_len - 1) as f64);
            let idx = center.floor() as u32;
            let frac = center - idx as f64;
            let w = (frac * 256.0).round() as u32;
            let idx_clamped = idx.min(src_len - 2);
            let w_final = if idx_clamped < idx { 256 } else { w.min(256) };
            AxisWeight { src_idx: idx_clamped, weight_x256: w_final as u16 }
        })
        .collect()
}

/// Split an `AxisWeight` table into the two flat `u32` arrays the GPU kernel
/// expects (one for src indices, one for weights). The kernel works in u32
/// space throughout to keep the fixed-point math identical to the cubecl
/// reference (which also widens to u32 before multiply).
pub fn split_weights(table: &[AxisWeight]) -> (Vec<u32>, Vec<u32>) {
    let idx = table.iter().map(|w| w.src_idx).collect();
    let w = table.iter().map(|w| w.weight_x256 as u32).collect();
    (idx, w)
}
