//! Fusion bench mirroring `crates/kornia-cubecl/examples/bench_fusion.rs`.
//!
//! Three pipelines:
//!  - P0: bilinear resize only (HWC u8 RGB out)
//!  - P1: resize → RGB→gray → normalize to f32 (HWC f32 gray out)
//!  - P2: resize → per-channel normalize → CHW f32 (model-ready tensor)
//!
//! For P1/P2 we time both the sequential multi-kernel path (with DRAM
//! intermediates) and the fused single-kernel path. The headline result
//! from the cubecl prototype was that fused P1 ≈ P0 cost — "adding
//! gray+norm is FREE" because the kernel is bandwidth-bound. We're
//! testing whether the same finding reproduces under cuda-oxide's
//! AOT LLVM-21 nvptx codegen.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_launch;
use kornia_cuda_oxide::*;
use kornia_cuda_oxide::weights::{compute_axis_weights, split_weights};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::time::Instant;

// (src_w, src_h, dst_w, dst_h) — same 5 sizes as kornia-cubecl bench_fusion
const PIPELINES: &[(u32, u32, u32, u32)] = &[
    (1024, 512, 512, 256),
    (2048, 1024, 1024, 512),
    (4096, 2048, 2048, 1024),
    (8192, 4096, 4096, 2048),
    (1920, 1080, 960, 540),
];
const REPS: usize = 10;
const WARMUP: usize = 3;
const MEAN: f32 = 127.5;
const INV_STD: f32 = 1.0 / 64.0;
// ImageNet-style per-channel mean/std (in u8 scale, 0-255)
const MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
const INV_STD_RGB: [f32; 3] = [1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375];

fn make_image_bytes(w: u32, h: u32) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut buf = vec![0u8; (w * h * 3) as usize];
    rng.fill_bytes(&mut buf);
    buf
}

fn stats(mut s: Vec<f64>) -> (f64, f64, f64) {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() / 2], s.iter().sum::<f64>() / s.len() as f64)
}

fn fmt_us(s: f64) -> String {
    format!("{:>9.1}", s * 1e6)
}
fn fmt_mpix(p: u64, s: f64) -> String {
    format!("{:>7.1}", (p as f64) / s / 1e6)
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let ptx_path = format!("{}/kornia_cuda_oxide.ptx", env!("CARGO_MANIFEST_DIR"));
    let module = ctx
        .load_module_from_file(&ptx_path)
        .expect("load_module_from_file");

    println!("\n# cuda-oxide pipeline fusion bench (GTX 1650, sm_75)");
    println!("# Reps={REPS}, warmup={WARMUP}\n");

    let stream = ctx.default_stream();

    // ---- Pipeline 0 baseline ----
    println!("## Pipeline 0: bilinear resize ONLY (HWC u8 RGB out)");
    println!(
        "{:<22}{:<18}{:>10}{:>10}{:>10}{:>10}",
        "src→dst", "arm", "min(μs)", "med(μs)", "mean(μs)", "Mpix/s"
    );
    println!("{}", "-".repeat(82));
    for &(src_w, src_h, dst_w, dst_h) in PIPELINES {
        let id = format!("{src_w}x{src_h}→{dst_w}x{dst_h}");
        let dst_pix = (dst_w as u64) * (dst_h as u64);
        let src_bytes = make_image_bytes(src_w, src_h);

        let wx = compute_axis_weights(src_w, dst_w);
        let wy = compute_axis_weights(src_h, dst_h);
        let (wx_idx, wx_w) = split_weights(&wx);
        let (wy_idx, wy_w) = split_weights(&wy);

        let src_dev = DeviceBuffer::from_host(&stream, &src_bytes).unwrap();
        let mut dst_dev =
            DeviceBuffer::<u8>::zeroed(&stream, (dst_w * dst_h * 3) as usize).unwrap();
        let wx_idx_dev = DeviceBuffer::from_host(&stream, &wx_idx).unwrap();
        let wx_w_dev = DeviceBuffer::from_host(&stream, &wx_w).unwrap();
        let wy_idx_dev = DeviceBuffer::from_host(&stream, &wy_idx).unwrap();
        let wy_w_dev = DeviceBuffer::from_host(&stream, &wy_w).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (dst_w.div_ceil(16), dst_h.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        for _ in 0..WARMUP {
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_dev),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h]
            }.unwrap();
            stream.synchronize().unwrap();
        }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_dev),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h]
            }.unwrap();
            stream.synchronize().unwrap();
            s.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(s);
        println!(
            "{:<22}{:<18}{}{}{}{}",
            id, "resize_only",
            fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md),
        );
    }

    // ---- Pipeline 1: resize → gray → normalize ----
    println!("\n## Pipeline 1: bilinear resize → RGB→gray → normalize_to_f32 (HWC out)");
    println!("# sequential = 3 kernel launches with DRAM intermediates");
    println!("# fused      = 1 kernel launch\n");
    println!(
        "{:<22}{:<18}{:>10}{:>10}{:>10}{:>10}",
        "src→dst", "arm", "min(μs)", "med(μs)", "mean(μs)", "Mpix/s"
    );
    println!("{}", "-".repeat(82));
    for &(src_w, src_h, dst_w, dst_h) in PIPELINES {
        let id = format!("{src_w}x{src_h}→{dst_w}x{dst_h}");
        let dst_pix = (dst_w as u64) * (dst_h as u64);
        let src_bytes = make_image_bytes(src_w, src_h);

        let wx = compute_axis_weights(src_w, dst_w);
        let wy = compute_axis_weights(src_h, dst_h);
        let (wx_idx, wx_w) = split_weights(&wx);
        let (wy_idx, wy_w) = split_weights(&wy);

        let src_dev = DeviceBuffer::from_host(&stream, &src_bytes).unwrap();
        let mut dst_resized =
            DeviceBuffer::<u8>::zeroed(&stream, (dst_w * dst_h * 3) as usize).unwrap();
        let mut dst_gray =
            DeviceBuffer::<u8>::zeroed(&stream, (dst_w * dst_h) as usize).unwrap();
        let mut dst_norm =
            DeviceBuffer::<f32>::zeroed(&stream, (dst_w * dst_h) as usize).unwrap();
        let mut dst_fused =
            DeviceBuffer::<f32>::zeroed(&stream, (dst_w * dst_h) as usize).unwrap();
        let wx_idx_dev = DeviceBuffer::from_host(&stream, &wx_idx).unwrap();
        let wx_w_dev = DeviceBuffer::from_host(&stream, &wx_w).unwrap();
        let wy_idx_dev = DeviceBuffer::from_host(&stream, &wy_idx).unwrap();
        let wy_w_dev = DeviceBuffer::from_host(&stream, &wy_w).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (dst_w.div_ceil(16), dst_h.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        // sequential 3k
        for _ in 0..WARMUP {
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_resized),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h]
            }.unwrap();
            cuda_launch! {
                kernel: rgb_to_gray_u8_kernel,
                stream: stream, module: module, config: cfg,
                args: [slice(dst_resized), slice_mut(dst_gray), dst_w, dst_h]
            }.unwrap();
            cuda_launch! {
                kernel: normalize_u8_to_f32_kernel,
                stream: stream, module: module, config: cfg,
                args: [slice(dst_gray), slice_mut(dst_norm), dst_w, dst_h, MEAN, INV_STD]
            }.unwrap();
            stream.synchronize().unwrap();
        }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_resized),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h]
            }.unwrap();
            cuda_launch! {
                kernel: rgb_to_gray_u8_kernel,
                stream: stream, module: module, config: cfg,
                args: [slice(dst_resized), slice_mut(dst_gray), dst_w, dst_h]
            }.unwrap();
            cuda_launch! {
                kernel: normalize_u8_to_f32_kernel,
                stream: stream, module: module, config: cfg,
                args: [slice(dst_gray), slice_mut(dst_norm), dst_w, dst_h, MEAN, INV_STD]
            }.unwrap();
            stream.synchronize().unwrap();
            s.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(s);
        println!(
            "{:<22}{:<18}{}{}{}{}",
            id, "sequential_3k",
            fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md),
        );

        // fused 1k
        for _ in 0..WARMUP {
            cuda_launch! {
                kernel: resize_to_gray_normalize,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_fused),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h, MEAN, INV_STD]
            }.unwrap();
            stream.synchronize().unwrap();
        }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            cuda_launch! {
                kernel: resize_to_gray_normalize,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_fused),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h, MEAN, INV_STD]
            }.unwrap();
            stream.synchronize().unwrap();
            s.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(s);
        println!(
            "{:<22}{:<18}{}{}{}{}",
            "", "fused_1k",
            fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md),
        );
        println!();
    }

    // ---- Pipeline 2: resize → per-channel normalize → CHW f32 ----
    println!("## Pipeline 2: bilinear resize → per-channel normalize → CHW f32 (ML preprocessing)");
    println!("# sequential = 2 kernel launches (resize HWC u8 + transpose+normalize to CHW f32)");
    println!("# fused      = 1 kernel launch (resize+normalize+CHW write all inline)\n");
    println!(
        "{:<22}{:<18}{:>10}{:>10}{:>10}{:>10}",
        "src→dst", "arm", "min(μs)", "med(μs)", "mean(μs)", "Mpix/s"
    );
    println!("{}", "-".repeat(82));
    for &(src_w, src_h, dst_w, dst_h) in PIPELINES {
        let id = format!("{src_w}x{src_h}→{dst_w}x{dst_h}");
        let dst_pix = (dst_w as u64) * (dst_h as u64);
        let src_bytes = make_image_bytes(src_w, src_h);

        let wx = compute_axis_weights(src_w, dst_w);
        let wy = compute_axis_weights(src_h, dst_h);
        let (wx_idx, wx_w) = split_weights(&wx);
        let (wy_idx, wy_w) = split_weights(&wy);

        let src_dev = DeviceBuffer::from_host(&stream, &src_bytes).unwrap();
        let mut dst_resized =
            DeviceBuffer::<u8>::zeroed(&stream, (dst_w * dst_h * 3) as usize).unwrap();
        let mut dst_chw_seq =
            DeviceBuffer::<f32>::zeroed(&stream, (dst_w * dst_h * 3) as usize).unwrap();
        let mut dst_chw_fused =
            DeviceBuffer::<f32>::zeroed(&stream, (dst_w * dst_h * 3) as usize).unwrap();
        let wx_idx_dev = DeviceBuffer::from_host(&stream, &wx_idx).unwrap();
        let wx_w_dev = DeviceBuffer::from_host(&stream, &wx_w).unwrap();
        let wy_idx_dev = DeviceBuffer::from_host(&stream, &wy_idx).unwrap();
        let wy_w_dev = DeviceBuffer::from_host(&stream, &wy_w).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (dst_w.div_ceil(16), dst_h.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        // sequential 2k
        for _ in 0..WARMUP {
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_resized),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h]
            }.unwrap();
            cuda_launch! {
                kernel: hwc_u8_to_chw_f32_normalize_kernel,
                stream: stream, module: module, config: cfg,
                args: [slice(dst_resized), slice_mut(dst_chw_seq), dst_w, dst_h,
                       MEAN_RGB[0], MEAN_RGB[1], MEAN_RGB[2],
                       INV_STD_RGB[0], INV_STD_RGB[1], INV_STD_RGB[2]]
            }.unwrap();
            stream.synchronize().unwrap();
        }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            cuda_launch! {
                kernel: resize_bilinear_u8_rgb,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_resized),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h]
            }.unwrap();
            cuda_launch! {
                kernel: hwc_u8_to_chw_f32_normalize_kernel,
                stream: stream, module: module, config: cfg,
                args: [slice(dst_resized), slice_mut(dst_chw_seq), dst_w, dst_h,
                       MEAN_RGB[0], MEAN_RGB[1], MEAN_RGB[2],
                       INV_STD_RGB[0], INV_STD_RGB[1], INV_STD_RGB[2]]
            }.unwrap();
            stream.synchronize().unwrap();
            s.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(s);
        println!(
            "{:<22}{:<18}{}{}{}{}",
            id, "sequential_2k",
            fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md),
        );

        // fused 1k
        for _ in 0..WARMUP {
            cuda_launch! {
                kernel: resize_to_chw_normalize,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_chw_fused),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h,
                       MEAN_RGB[0], MEAN_RGB[1], MEAN_RGB[2],
                       INV_STD_RGB[0], INV_STD_RGB[1], INV_STD_RGB[2]]
            }.unwrap();
            stream.synchronize().unwrap();
        }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            cuda_launch! {
                kernel: resize_to_chw_normalize,
                stream: stream, module: module, config: cfg,
                args: [slice(src_dev), slice_mut(dst_chw_fused),
                       slice(wx_idx_dev), slice(wx_w_dev),
                       slice(wy_idx_dev), slice(wy_w_dev),
                       src_w, dst_w, dst_h,
                       MEAN_RGB[0], MEAN_RGB[1], MEAN_RGB[2],
                       INV_STD_RGB[0], INV_STD_RGB[1], INV_STD_RGB[2]]
            }.unwrap();
            stream.synchronize().unwrap();
            s.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(s);
        println!(
            "{:<22}{:<18}{}{}{}{}",
            "", "fused_1k",
            fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md),
        );
        println!();
    }
}
