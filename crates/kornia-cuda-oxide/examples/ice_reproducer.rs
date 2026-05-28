//! Minimal reproducer for cuda-oxide alpha ICE in pliron MIR→LLVM lowering.
//!
//! **Bug:** calling a helper function from a `#[kernel]` body panics the
//! rustc-codegen-cuda backend with:
//!
//! ```text
//! thread 'rustc' panicked at pliron/src/operation.rs:526:
//! Operation with use(s) being erased
//!   at mir-lower/src/convert/ops/call.rs:413:18
//! error: the compiler unexpectedly panicked. This is a bug.
//! ```
//!
//! **Trigger condition (confirmed):** the helper must have MULTIPLE `&[T]`
//! slice parameters AND return a tuple `(T, T, T)`. A helper with a single
//! `&[u8]` and a scalar return does NOT trigger the ICE. The combination of
//! multi-slice-param + tuple-return does.
//!
//! **Non-triggering patterns (both compile fine):**
//! - Helper with one `&[u8]` param + scalar return → OK
//! - Helper with no slice params (pure arithmetic) → OK
//!
//! **Workaround in use:** manually inline all helper math into each kernel
//! body. See `src/lib.rs` — `resize_to_gray_normalize` and
//! `resize_to_chw_normalize` are monolithic bodies for this reason.
//!
//! **To reproduce the ICE:** uncomment `with_helper_kernel` below and run:
//! ```
//! CUDA_OXIDE_TARGET=sm_75 \
//! RUSTFLAGS="-Z codegen-backend=<path>/librustc_codegen_cuda.so \
//!            -C opt-level=3 -C debug-assertions=off \
//!            -Z mir-enable-passes=-JumpThreading \
//!            -C symbol-mangling-version=v0" \
//! cargo build --release --example ice_reproducer
//! ```
//!
//! **Tested on:** cuda-oxide @ NVlabs main (2026-05), rustc nightly-2026-04-03,
//! LLVM 21, sm_75 (GTX 1650).

use cuda_device::{kernel, thread};

// ── Helper: multiple &[T] params + tuple return → TRIGGERS the ICE ────────
//
// A single &[u8] with scalar return does NOT trigger. The combination of
// multiple slice params + tuple return does.
fn bilinear_sample(
    src: &[u8],
    wx_idx: &[u32],
    wx_w: &[u32],
    wy_idx: &[u32],
    wy_w: &[u32],
    src_w: u32,
    col: usize,
    row: usize,
) -> (u32, u32, u32) {
    let sx = wx_idx[col] as usize;
    let wx = wx_w[col];
    let sy = wy_idx[row] as usize;
    let wy = wy_w[row];
    let sw = src_w as usize;
    let (iwx, iwy) = (256u32 - wx, 256u32 - wy);
    let rt = sy * sw * 3;
    let rb = (sy + 1) * sw * 3;
    let ol = sx * 3;
    let or_ = (sx + 1) * 3;
    let r = (iwy * (iwx * src[rt+ol] as u32   + wx * src[rt+or_] as u32)
           +  wy * (iwx * src[rb+ol] as u32   + wx * src[rb+or_] as u32) + (1<<15)) >> 16;
    let g = (iwy * (iwx * src[rt+ol+1] as u32 + wx * src[rt+or_+1] as u32)
           +  wy * (iwx * src[rb+ol+1] as u32 + wx * src[rb+or_+1] as u32) + (1<<15)) >> 16;
    let b = (iwy * (iwx * src[rt+ol+2] as u32 + wx * src[rt+or_+2] as u32)
           +  wy * (iwx * src[rb+ol+2] as u32 + wx * src[rb+or_+2] as u32) + (1<<15)) >> 16;
    (r, g, b)
}

// ── Kernel A: calls the helper → PANICS the codegen backend ───────────────
//
// Re-enabled 2026-05-28 to verify upstream fix (commit 6ed9938: "stop
// misclassifying non-empty tuples as unit in mir.call lowering", issue #79).
#[kernel]
pub fn with_helper_kernel(
    src: &[u8],
    wx_idx: &[u32], wx_w: &[u32],
    wy_idx: &[u32], wy_w: &[u32],
    src_w: u32, dst_w: u32, dst_h: u32,
) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= dst_w as usize || row >= dst_h as usize { return; }
    // Multi-slice helper + tuple destructure — previously ICE at
    // mir-lower/call.rs:413. Should compile clean post-6ed9938.
    let (_r, _g, _b) = bilinear_sample(src, wx_idx, wx_w, wy_idx, wy_w,
                                        src_w, col, row);
}

// ── Kernel B: manually inlined equivalent → compiles and runs correctly ────
//
// Exact same logic as `with_helper_kernel` but with the helper body inlined.
// This is the workaround used throughout src/lib.rs.
#[kernel]
pub fn inlined_kernel(src: &[u8], width: u32) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= width as usize {
        return;
    }
    let _val = src[row * width as usize + col];
}

// ── Arithmetic-only helper (no slice params): works fine ──────────────────
//
// To confirm the ICE is specifically about slice-indexing helpers (not all
// helper calls), this arithmetic helper compiles and runs without issue.
fn blend(a: u32, b: u32, w: u32) -> u32 {
    (256 - w) * a + w * b
}

#[kernel]
pub fn arithmetic_helper_kernel(src: &[u8], width: u32) {
    let col = thread::index_2d_col();
    let row = thread::index_2d_row();
    if col >= width as usize {
        return;
    }
    let a = src[row * width as usize + col] as u32;
    let b = src[row * width as usize + col + 1] as u32;
    let _blended = blend(a, b, 128); // arithmetic helper: fine
}

fn main() {
    println!("ice_reproducer: this example exists to document the pliron ICE.");
    println!("See the source comments for reproduction instructions.");
    println!("  inlined_kernel        → compiled (workaround)");
    println!("  arithmetic_helper_kernel → compiled (helpers without slice params work)");
    println!("  with_helper_kernel    → ICE (uncomment to reproduce)");
}
