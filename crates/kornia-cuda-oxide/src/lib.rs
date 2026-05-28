//! kornia-cuda-oxide — cuda-oxide GPU kernels for kornia-rs (prototype).
//!
//! Goal-5 bake-off counterpart to `crates/kornia-cubecl/`. Implements the same
//! kernels (starting with bilinear u8 RGB resize) on the cuda-oxide rustc
//! codegen backend so we can compare integration complexity, safety surface,
//! ergonomics, and perf head-to-head before locking the kornia GPU backend.
//!
//! Bit-exact correctness against `fast_image_resize` is asserted in
//! `examples/correctness.rs`.
//!
//! # Kernel-loading pattern
//!
//! Uses cuda-oxide's **free `#[kernel]` + `load_module_from_file`** path
//! (same as the `hashmap_v2` upstream example) rather than `#[cuda_module]`.
//! Rationale: `#[cuda_module]`'s `kernels::load(ctx)` mechanism embeds the
//! PTX into the binary via an unreferenced `.oxart` section that the
//! linker strips when the kernel module lives in a library crate consumed
//! by separate binaries (examples, tests, benches). The free-`#[kernel]`
//! path emits `kornia_cuda_oxide.ptx` as a sibling file at build time, and
//! the host loads it explicitly — works cleanly across multiple consumer
//! binaries that share this crate.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_launch;
use seq_macro::seq;
use std::sync::Arc;

pub mod weights;

use weights::{compute_axis_weights, split_weights};

// =============================================================================
// KERNEL — compiled to PTX by rustc-codegen-cuda, written to
// `kornia_cuda_oxide.ptx` alongside the build artifact.
// =============================================================================

/// Bilinear u8 RGB resize. One thread per output pixel.
///
/// Mirrors `kornia_cubecl::resize::resize_bilinear_u8_rgb_kernel`. Same
/// fixed-point math, same weight table layout — bit-exact equivalence is
/// asserted by `examples/correctness.rs`.
///
/// `weights_x_*` is length `dst_w`, `weights_y_*` is length `dst_h`.
/// `dst` is `dst_w * dst_h * 3` u8 bytes in HWC RGB layout.
#[kernel]
pub fn resize_bilinear_u8_rgb(
    src: &[u8],
    mut dst: DisjointSlice<u8, thread::Runtime2DIndex>,
    weights_x_idx: &[u32],
    weights_x_w: &[u32],
    weights_y_idx: &[u32],
    weights_y_w: &[u32],
    src_w: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let row = thread::index_2d_row();
    let col = thread::index_2d_col();

    // SAFETY: every thread in this launch passes the same `dst_w` as the
    // row stride. `index_2d_runtime` returns Some only when `col < dst_w`,
    // i.e. our active threads cover disjoint columns; combined with the
    // separate `row < dst_h` check below, every surviving thread targets
    // a unique output pixel.
    if let Some(_dst_idx) = unsafe { thread::index_2d_runtime(dst_w as usize) }
        && row < dst_h as usize
    {
        let sx = weights_x_idx[col] as usize;
        let wx = weights_x_w[col];
        let sy = weights_y_idx[row] as usize;
        let wy = weights_y_w[row];

        let src_w_us = src_w as usize;
        let row_top = sy * src_w_us * 3;
        let row_bot = (sy + 1) * src_w_us * 3;
        let off_l = sx * 3;
        let off_r = (sx + 1) * 3;

        let inv_wx = 256u32 - wx;
        let inv_wy = 256u32 - wy;

        let dst_off = (row * (dst_w as usize) + col) * 3;

        // Channel R (offset +0)
        let tl_r = src[row_top + off_l] as u32;
        let tr_r = src[row_top + off_r] as u32;
        let bl_r = src[row_bot + off_l] as u32;
        let br_r = src[row_bot + off_r] as u32;
        let top_r = inv_wx * tl_r + wx * tr_r;
        let bot_r = inv_wx * bl_r + wx * br_r;
        let r_val = ((inv_wy * top_r + wy * bot_r + (1u32 << 15)) >> 16) as u8;

        // Channel G (offset +1)
        let tl_g = src[row_top + off_l + 1] as u32;
        let tr_g = src[row_top + off_r + 1] as u32;
        let bl_g = src[row_bot + off_l + 1] as u32;
        let br_g = src[row_bot + off_r + 1] as u32;
        let top_g = inv_wx * tl_g + wx * tr_g;
        let bot_g = inv_wx * bl_g + wx * br_g;
        let g_val = ((inv_wy * top_g + wy * bot_g + (1u32 << 15)) >> 16) as u8;

        // Channel B (offset +2)
        let tl_b = src[row_top + off_l + 2] as u32;
        let tr_b = src[row_top + off_r + 2] as u32;
        let bl_b = src[row_bot + off_l + 2] as u32;
        let br_b = src[row_bot + off_r + 2] as u32;
        let top_b = inv_wx * tl_b + wx * tr_b;
        let bot_b = inv_wx * bl_b + wx * br_b;
        let b_val = ((inv_wy * top_b + wy * bot_b + (1u32 << 15)) >> 16) as u8;

        // SAFETY: `(row, col)` are bounded by `(dst_h, dst_w)`, so
        // `dst_off + {0,1,2} < dst_w * dst_h * 3` (the slice length).
        // Each surviving thread writes a unique 3-byte tuple — no overlap.
        unsafe {
            *dst.get_unchecked_mut(dst_off) = r_val;
            *dst.get_unchecked_mut(dst_off + 1) = g_val;
            *dst.get_unchecked_mut(dst_off + 2) = b_val;
        }
    }
}

// =============================================================================
// TILED VARIANTS — each thread writes N horizontally-adjacent dst pixels.
//
// These mirror `kornia_cubecl::resize::resize_bilinear_u8_rgb_kernel_x{4,16}`.
// Reduces total thread count by N×, exposing longer contiguous-byte-store
// patterns to the LLVM-21 nvptx backend's vectorization passes.
//
// Launch geometry: `dst_w` must be divisible by N. Caller is responsible.
// =============================================================================

/// Tiled bilinear u8 RGB resize — 4 dst pixels per thread (12 bytes contiguous).
/// `dst_w` MUST be divisible by 4. Same fixed-point math, bit-exact same output
/// as `resize_bilinear_u8_rgb`.
#[kernel]
pub fn resize_bilinear_u8_rgb_x4(
    src: &[u8],
    mut dst: DisjointSlice<u8>,
    weights_x_idx: &[u32],
    weights_x_w: &[u32],
    weights_y_idx: &[u32],
    weights_y_w: &[u32],
    src_w: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let tile_x = thread::index_2d_col();
    let row = thread::index_2d_row();
    if tile_x * 4 >= dst_w as usize || row >= dst_h as usize {
        return;
    }

    let sy = weights_y_idx[row] as usize;
    let wy = weights_y_w[row];
    let inv_wy = 256u32 - wy;

    let src_w_us = src_w as usize;
    let row_top = sy * src_w_us * 3;
    let row_bot = (sy + 1) * src_w_us * 3;
    let dst_row = row * (dst_w as usize) * 3;
    let base_x = tile_x * 4;

    seq!(PX in 0..4 {
        {
            let out_x = base_x + PX;
            let sx = weights_x_idx[out_x] as usize;
            let wx = weights_x_w[out_x];
            let inv_wx = 256u32 - wx;
            let off_l = sx * 3;
            let off_r = (sx + 1) * 3;
            let dst_off = dst_row + out_x * 3;
            seq!(CH in 0..3 {
                {
                    let tl = src[row_top + off_l + CH] as u32;
                    let tr = src[row_top + off_r + CH] as u32;
                    let bl = src[row_bot + off_l + CH] as u32;
                    let br = src[row_bot + off_r + CH] as u32;
                    let top = inv_wx * tl + wx * tr;
                    let bot = inv_wx * bl + wx * br;
                    let val = ((inv_wy * top + wy * bot + (1u32 << 15)) >> 16) as u8;
                    // SAFETY: out_x ∈ [base_x, base_x+4), base_x*4 < dst_w (precondition
                    // dst_w % 4 == 0 + tile_x*4 < dst_w). dst_off + CH < dst_w*dst_h*3.
                    unsafe { *dst.get_unchecked_mut(dst_off + CH) = val; }
                }
            });
        }
    });
}

/// Tiled bilinear u8 RGB resize — 16 dst pixels per thread (48 bytes contiguous).
/// `dst_w` MUST be divisible by 16. Same fixed-point math, bit-exact same output
/// as `resize_bilinear_u8_rgb`.
#[kernel]
pub fn resize_bilinear_u8_rgb_x16(
    src: &[u8],
    mut dst: DisjointSlice<u8>,
    weights_x_idx: &[u32],
    weights_x_w: &[u32],
    weights_y_idx: &[u32],
    weights_y_w: &[u32],
    src_w: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let tile_x = thread::index_2d_col();
    let row = thread::index_2d_row();
    if tile_x * 16 >= dst_w as usize || row >= dst_h as usize {
        return;
    }

    let sy = weights_y_idx[row] as usize;
    let wy = weights_y_w[row];
    let inv_wy = 256u32 - wy;

    let src_w_us = src_w as usize;
    let row_top = sy * src_w_us * 3;
    let row_bot = (sy + 1) * src_w_us * 3;
    let dst_row = row * (dst_w as usize) * 3;
    let base_x = tile_x * 16;

    seq!(PX in 0..16 {
        {
            let out_x = base_x + PX;
            let sx = weights_x_idx[out_x] as usize;
            let wx = weights_x_w[out_x];
            let inv_wx = 256u32 - wx;
            let off_l = sx * 3;
            let off_r = (sx + 1) * 3;
            let dst_off = dst_row + out_x * 3;
            seq!(CH in 0..3 {
                {
                    let tl = src[row_top + off_l + CH] as u32;
                    let tr = src[row_top + off_r + CH] as u32;
                    let bl = src[row_bot + off_l + CH] as u32;
                    let br = src[row_bot + off_r + CH] as u32;
                    let top = inv_wx * tl + wx * tr;
                    let bot = inv_wx * bl + wx * br;
                    let val = ((inv_wy * top + wy * bot + (1u32 << 15)) >> 16) as u8;
                    // SAFETY: as above, with stride 16 instead of 4.
                    unsafe { *dst.get_unchecked_mut(dst_off + CH) = val; }
                }
            });
        }
    });
}

// =============================================================================
// FUSED PIPELINES — the headline test of cuda-oxide's AOT codegen quality
// vs cubecl's JIT MLIR codegen. Mirror `kornia_cubecl::resize::resize_to_*`
// kernels exactly. Helpers (sample / gray / normalize) inline at codegen.
// =============================================================================

// Helpers are inlined MANUALLY into each fused kernel below — cuda-oxide
// 0.1-alpha ICE's its MIR→LLVM call-lowering pass when a `#[kernel]` calls
// a helper fn that indexes a slice param (panics: "Operation with use(s)
// being erased" at mir-lower/src/convert/ops/call.rs:413). Tried
// `#[inline(always)]`, `#[device]`, and combinations — same ICE. cubecl's
// `#[cube]` primitives have no equivalent issue because cubecl inlines at
// the cubecl IR level, before any LLVM-bound MIR conversion.
//
// Each fused kernel is therefore one monolithic body. Documented as an
// ergonomic disadvantage in COMPARISON.md.

/// FUSED: resize bilinear u8 RGB + RGB→gray + normalize to f32, all in one
/// kernel. Output is HWC f32, one channel (gray), `dst_w * dst_h` floats.
#[kernel]
pub fn resize_to_gray_normalize(
    src: &[u8],
    mut dst: DisjointSlice<f32>,
    weights_x_idx: &[u32],
    weights_x_w: &[u32],
    weights_y_idx: &[u32],
    weights_y_w: &[u32],
    src_w: u32,
    dst_w: u32,
    dst_h: u32,
    mean: f32,
    inv_std: f32,
) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= dst_w as usize || row >= dst_h as usize {
        return;
    }

    // Inlined: bilinear sample
    let sx = weights_x_idx[col] as usize;
    let wx = weights_x_w[col];
    let sy = weights_y_idx[row] as usize;
    let wy = weights_y_w[row];
    let src_w_us = src_w as usize;
    let row_top = sy * src_w_us * 3;
    let row_bot = (sy + 1) * src_w_us * 3;
    let off_l = sx * 3;
    let off_r = (sx + 1) * 3;
    let inv_wx = 256u32 - wx;
    let inv_wy = 256u32 - wy;

    let tl_r = src[row_top + off_l] as u32;
    let tr_r = src[row_top + off_r] as u32;
    let bl_r = src[row_bot + off_l] as u32;
    let br_r = src[row_bot + off_r] as u32;
    let r_val = ((inv_wy * (inv_wx * tl_r + wx * tr_r) + wy * (inv_wx * bl_r + wx * br_r) + (1u32 << 15)) >> 16);

    let tl_g = src[row_top + off_l + 1] as u32;
    let tr_g = src[row_top + off_r + 1] as u32;
    let bl_g = src[row_bot + off_l + 1] as u32;
    let br_g = src[row_bot + off_r + 1] as u32;
    let g_val = ((inv_wy * (inv_wx * tl_g + wx * tr_g) + wy * (inv_wx * bl_g + wx * br_g) + (1u32 << 15)) >> 16);

    let tl_b = src[row_top + off_l + 2] as u32;
    let tr_b = src[row_top + off_r + 2] as u32;
    let bl_b = src[row_bot + off_l + 2] as u32;
    let br_b = src[row_bot + off_r + 2] as u32;
    let b_val = ((inv_wy * (inv_wx * tl_b + wx * tr_b) + wy * (inv_wx * bl_b + wx * br_b) + (1u32 << 15)) >> 16);

    // Inlined: rgb_to_gray_u8 (BT.601 fixed-point)
    let gray = (r_val * 77 + g_val * 150 + b_val * 29 + 128) >> 8;
    // Inlined: normalize_u8_to_f32
    let norm = (gray as f32 - mean) * inv_std;

    let dst_off = row * (dst_w as usize) + col;
    // SAFETY: row < dst_h && col < dst_w, so dst_off < dst_w * dst_h = slice len.
    unsafe {
        *dst.get_unchecked_mut(dst_off) = norm;
    }
}

/// FUSED: resize + per-channel normalize → CHW f32 (model-ready tensor).
/// Output layout: `dst[c * dst_h * dst_w + y * dst_w + x]` for c ∈ {0=R, 1=G, 2=B}.
#[kernel]
pub fn resize_to_chw_normalize(
    src: &[u8],
    mut dst: DisjointSlice<f32>,
    weights_x_idx: &[u32],
    weights_x_w: &[u32],
    weights_y_idx: &[u32],
    weights_y_w: &[u32],
    src_w: u32,
    dst_w: u32,
    dst_h: u32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    inv_std_r: f32,
    inv_std_g: f32,
    inv_std_b: f32,
) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= dst_w as usize || row >= dst_h as usize {
        return;
    }

    // Inlined: bilinear sample (identical math to fused above)
    let sx = weights_x_idx[col] as usize;
    let wx = weights_x_w[col];
    let sy = weights_y_idx[row] as usize;
    let wy = weights_y_w[row];
    let src_w_us = src_w as usize;
    let row_top = sy * src_w_us * 3;
    let row_bot = (sy + 1) * src_w_us * 3;
    let off_l = sx * 3;
    let off_r = (sx + 1) * 3;
    let inv_wx = 256u32 - wx;
    let inv_wy = 256u32 - wy;

    let tl_r = src[row_top + off_l] as u32;
    let tr_r = src[row_top + off_r] as u32;
    let bl_r = src[row_bot + off_l] as u32;
    let br_r = src[row_bot + off_r] as u32;
    let r_val = ((inv_wy * (inv_wx * tl_r + wx * tr_r) + wy * (inv_wx * bl_r + wx * br_r) + (1u32 << 15)) >> 16);

    let tl_g = src[row_top + off_l + 1] as u32;
    let tr_g = src[row_top + off_r + 1] as u32;
    let bl_g = src[row_bot + off_l + 1] as u32;
    let br_g = src[row_bot + off_r + 1] as u32;
    let g_val = ((inv_wy * (inv_wx * tl_g + wx * tr_g) + wy * (inv_wx * bl_g + wx * br_g) + (1u32 << 15)) >> 16);

    let tl_b = src[row_top + off_l + 2] as u32;
    let tr_b = src[row_top + off_r + 2] as u32;
    let bl_b = src[row_bot + off_l + 2] as u32;
    let br_b = src[row_bot + off_r + 2] as u32;
    let b_val = ((inv_wy * (inv_wx * tl_b + wx * tr_b) + wy * (inv_wx * bl_b + wx * br_b) + (1u32 << 15)) >> 16);

    let plane = (dst_h as usize) * (dst_w as usize);
    let off = row * (dst_w as usize) + col;
    // SAFETY: dst is dst_w*dst_h*3 f32s. off + 2*plane < 3*plane = len.
    unsafe {
        *dst.get_unchecked_mut(off) = (r_val as f32 - mean_r) * inv_std_r;
        *dst.get_unchecked_mut(plane + off) = (g_val as f32 - mean_g) * inv_std_g;
        *dst.get_unchecked_mut(2 * plane + off) = (b_val as f32 - mean_b) * inv_std_b;
    }
}

// =============================================================================
// STANDALONE OPS — for the sequential bench arm (gray pipeline does
// resize → rgb_to_gray → normalize as 3 separate kernels with intermediate
// DRAM buffers; CHW pipeline does resize → hwc_to_chw_normalize as 2).
// =============================================================================

/// Standalone RGB → gray on an already-interleaved u8 RGB buffer.
#[kernel]
pub fn rgb_to_gray_u8_kernel(
    src: &[u8],
    mut dst: DisjointSlice<u8>,
    width: u32,
    height: u32,
) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= width as usize || row >= height as usize {
        return;
    }
    let off = row * (width as usize) + col;
    let r = src[off * 3] as u32;
    let g = src[off * 3 + 1] as u32;
    let b = src[off * 3 + 2] as u32;
    // Inlined: BT.601 fixed-point luma
    let gray = ((r * 77 + g * 150 + b * 29 + 128) >> 8) as u8;
    // SAFETY: off < width * height = slice len.
    unsafe { *dst.get_unchecked_mut(off) = gray; }
}

/// Standalone u8 → f32 normalize on a single-channel buffer.
#[kernel]
pub fn normalize_u8_to_f32_kernel(
    src: &[u8],
    mut dst: DisjointSlice<f32>,
    width: u32,
    height: u32,
    mean: f32,
    inv_std: f32,
) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= width as usize || row >= height as usize {
        return;
    }
    let off = row * (width as usize) + col;
    let val = (src[off] as f32 - mean) * inv_std;
    // SAFETY: off < width * height = slice len.
    unsafe { *dst.get_unchecked_mut(off) = val; }
}

/// Standalone HWC u8 RGB → CHW f32 with per-channel normalize. Two-kernel
/// sequential alternative to `resize_to_chw_normalize` (operates on an
/// already-resized HWC buffer).
#[kernel]
pub fn hwc_u8_to_chw_f32_normalize_kernel(
    src: &[u8],
    mut dst: DisjointSlice<f32>,
    width: u32,
    height: u32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    inv_std_r: f32,
    inv_std_g: f32,
    inv_std_b: f32,
) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= width as usize || row >= height as usize {
        return;
    }
    let plane = (height as usize) * (width as usize);
    let off = row * (width as usize) + col;
    let r = src[off * 3] as u32;
    let g = src[off * 3 + 1] as u32;
    let b = src[off * 3 + 2] as u32;
    // SAFETY: off + 2*plane < 3*plane = slice len.
    unsafe {
        *dst.get_unchecked_mut(off) = (r as f32 - mean_r) * inv_std_r;
        *dst.get_unchecked_mut(plane + off) = (g as f32 - mean_g) * inv_std_g;
        *dst.get_unchecked_mut(2 * plane + off) = (b as f32 - mean_b) * inv_std_b;
    }
}

// =============================================================================
// HOST LAUNCHER — convenience wrapper. Allocates device buffers, uploads, runs,
// downloads. For benchmarking the kernel in isolation, callers should reuse a
// pre-loaded module + pre-allocated buffers and time only the launch + sync.
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUDA error: {0}")]
    Cuda(String),
}

/// End-to-end resize: upload src, launch kernel, download dst. Convenient for
/// correctness tests; benches should split this into pre-allocated buffer + a
/// kernel-only timing path.
///
/// `ptx_path` is the filesystem path to the `kornia_cuda_oxide.ptx` artifact
/// produced by the cuda-oxide codegen backend at build time. For the bundled
/// examples this lives at `target/release/examples/kornia_cuda_oxide.ptx`
/// (cargo-oxide writes it alongside the binary).
pub fn resize_bilinear_u8_rgb_e2e(
    ctx: &Arc<CudaContext>,
    ptx_path: &str,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, Error> {
    assert_eq!(src.len(), (src_w * src_h * 3) as usize, "src length mismatch");

    let stream = ctx.default_stream();

    let wx = compute_axis_weights(src_w, dst_w);
    let wy = compute_axis_weights(src_h, dst_h);
    let (wx_idx, wx_w) = split_weights(&wx);
    let (wy_idx, wy_w) = split_weights(&wy);

    let src_dev = DeviceBuffer::from_host(&stream, src)
        .map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let mut dst_dev = DeviceBuffer::<u8>::zeroed(&stream, (dst_w * dst_h * 3) as usize)
        .map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let wx_idx_dev = DeviceBuffer::from_host(&stream, &wx_idx)
        .map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let wx_w_dev = DeviceBuffer::from_host(&stream, &wx_w)
        .map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let wy_idx_dev = DeviceBuffer::from_host(&stream, &wy_idx)
        .map_err(|e| Error::Cuda(format!("{e:?}")))?;
    let wy_w_dev = DeviceBuffer::from_host(&stream, &wy_w)
        .map_err(|e| Error::Cuda(format!("{e:?}")))?;

    let block = (16u32, 16u32, 1u32);
    let grid = (dst_w.div_ceil(block.0), dst_h.div_ceil(block.1), 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    let module = ctx
        .load_module_from_file(ptx_path)
        .map_err(|e| Error::Cuda(format!("load_module_from_file({ptx_path}): {e:?}")))?;

    cuda_launch! {
        kernel: resize_bilinear_u8_rgb,
        stream: stream,
        module: module,
        config: cfg,
        args: [
            slice(src_dev),
            slice_mut(dst_dev),
            slice(wx_idx_dev),
            slice(wx_w_dev),
            slice(wy_idx_dev),
            slice(wy_w_dev),
            src_w,
            dst_w,
            dst_h
        ]
    }
    .map_err(|e| Error::Cuda(format!("kernel launch: {e:?}")))?;

    dst_dev
        .to_host_vec(&stream)
        .map_err(|e| Error::Cuda(format!("{e:?}")))
}
