//! Correctness: cuda-oxide kernel must match `fast_image_resize` (x86 AVX2 path
//! on this machine) to within ±1 LSB per channel, ≤ 0.1% mismatched channels.
//!
//! Mirrors `crates/kornia-cubecl/tests/correctness.rs` exactly so the two
//! prototypes are held to the same correctness gate. The cubecl version
//! achieves max_diff=0 on Jetson; we're aiming for the same bit-exact result.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_launch;
use kornia_cuda_oxide::resize_bilinear_u8_rgb_e2e;
use kornia_cuda_oxide::weights::{compute_axis_weights, split_weights};
use kornia_cuda_oxide::*;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize};
use kornia_tensor::CpuAllocator;
use rand::{RngCore, SeedableRng, rngs::StdRng};

const SIZES: &[(u32, u32)] = &[(512, 256), (1024, 512), (2048, 1024), (4096, 2048)];
const TOLERANCE_LSB: u8 = 1;
const MAX_MISMATCH_FRAC: f64 = 0.001;

fn make_image_bytes(w: u32, h: u32) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut buf = vec![0u8; (w * h * 3) as usize];
    rng.fill_bytes(&mut buf);
    buf
}

fn cpu_reference(src_bytes: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let src = Image::<u8, 3, _>::new(
        ImageSize {
            width: src_w as usize,
            height: src_h as usize,
        },
        src_bytes.to_vec(),
        CpuAllocator,
    )
    .unwrap();
    let mut dst = Image::<u8, 3, _>::from_size_val(
        ImageSize {
            width: dst_w as usize,
            height: dst_h as usize,
        },
        0,
        CpuAllocator,
    )
    .unwrap();
    resize::resize_fast_rgb(&src, &mut dst, InterpolationMode::Bilinear).unwrap();
    dst.as_slice().to_vec()
}

fn compare(reference: &[u8], actual: &[u8], dst_w: u32, dst_h: u32, label: &str) {
    assert_eq!(
        reference.len(),
        actual.len(),
        "{label}: buffer length mismatch"
    );
    let total_channels = (dst_w as f64) * (dst_h as f64) * 3.0;
    let max_mismatch = (total_channels * MAX_MISMATCH_FRAC).ceil() as usize;
    let mut bad = 0usize;
    let mut max_diff: i32 = 0;
    for (r, a) in reference.iter().zip(actual.iter()) {
        let d = (*r as i32 - *a as i32).abs();
        if d > TOLERANCE_LSB as i32 {
            bad += 1;
        }
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        bad <= max_mismatch,
        "{label}: {bad} channels differ by > {TOLERANCE_LSB} LSB (max allowed {max_mismatch} of {total_channels}); max_diff={max_diff}"
    );
    eprintln!(
        "[{label}] {dst_w}x{dst_h}: bad={bad}/{}, max_diff={max_diff} (within tol)",
        total_channels as usize
    );
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    // The cuda-oxide codegen backend writes `kornia_cuda_oxide.ptx` to the
    // crate root (sibling of `src/`), not into target/. Resolve absolute
    // path at compile time via CARGO_MANIFEST_DIR so the binary loads
    // correctly regardless of CWD.
    let ptx_path = format!(
        "{}/kornia_cuda_oxide.ptx",
        env!("CARGO_MANIFEST_DIR")
    );

    println!("=== kornia-cuda-oxide correctness vs fast_image_resize ===");
    println!("    PTX module: {ptx_path}\n");

    let module = ctx
        .load_module_from_file(&ptx_path)
        .expect("load_module_from_file");

    for &(src_w, src_h) in SIZES {
        let (dst_w, dst_h) = (src_w / 2, src_h / 2);
        let src = make_image_bytes(src_w, src_h);
        let reference = cpu_reference(&src, src_w, src_h, dst_w, dst_h);

        // x1 baseline (uses the bundled launcher)
        let actual_x1 = resize_bilinear_u8_rgb_e2e(
            &ctx, &ptx_path, &src, src_w, src_h, dst_w, dst_h,
        )
        .expect("cuda-oxide resize failed");
        compare(
            &reference,
            &actual_x1,
            dst_w,
            dst_h,
            &format!("x1   {src_w}x{src_h}->{dst_w}x{dst_h}"),
        );

        // x4 and x16 — call kernels directly via cuda_launch! since the bundled
        // launcher only exposes x1.
        let stream = ctx.default_stream();
        let wx = compute_axis_weights(src_w, dst_w);
        let wy = compute_axis_weights(src_h, dst_h);
        let (wx_idx, wx_w) = split_weights(&wx);
        let (wy_idx, wy_w) = split_weights(&wy);
        let src_dev = DeviceBuffer::from_host(&stream, &src).unwrap();
        let wx_idx_dev = DeviceBuffer::from_host(&stream, &wx_idx).unwrap();
        let wx_w_dev = DeviceBuffer::from_host(&stream, &wx_w).unwrap();
        let wy_idx_dev = DeviceBuffer::from_host(&stream, &wy_idx).unwrap();
        let wy_w_dev = DeviceBuffer::from_host(&stream, &wy_w).unwrap();

        if dst_w % 4 == 0 {
            let mut dst_dev =
                DeviceBuffer::<u8>::zeroed(&stream, (dst_w * dst_h * 3) as usize).unwrap();
            let cfg = LaunchConfig {
                grid_dim: ((dst_w / 4).div_ceil(16), dst_h.div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb_x4,
                stream: stream,
                module: module,
                config: cfg,
                args: [
                    slice(src_dev), slice_mut(dst_dev),
                    slice(wx_idx_dev), slice(wx_w_dev),
                    slice(wy_idx_dev), slice(wy_w_dev),
                    src_w, dst_w, dst_h
                ]
            }
            .unwrap();
            let actual_x4 = dst_dev.to_host_vec(&stream).unwrap();
            compare(
                &reference,
                &actual_x4,
                dst_w,
                dst_h,
                &format!("x4   {src_w}x{src_h}->{dst_w}x{dst_h}"),
            );
        }

        if dst_w % 16 == 0 {
            let mut dst_dev =
                DeviceBuffer::<u8>::zeroed(&stream, (dst_w * dst_h * 3) as usize).unwrap();
            let cfg = LaunchConfig {
                grid_dim: ((dst_w / 16).div_ceil(16), dst_h.div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb_x16,
                stream: stream,
                module: module,
                config: cfg,
                args: [
                    slice(src_dev), slice_mut(dst_dev),
                    slice(wx_idx_dev), slice(wx_w_dev),
                    slice(wy_idx_dev), slice(wy_w_dev),
                    src_w, dst_w, dst_h
                ]
            }
            .unwrap();
            let actual_x16 = dst_dev.to_host_vec(&stream).unwrap();
            compare(
                &reference,
                &actual_x16,
                dst_w,
                dst_h,
                &format!("x16  {src_w}x{src_h}->{dst_w}x{dst_h}"),
            );
        }
    }

    println!("\n✓ All sizes / all variants passed correctness gate (≤ {TOLERANCE_LSB} LSB, ≤ 0.1% mismatches).");
}
