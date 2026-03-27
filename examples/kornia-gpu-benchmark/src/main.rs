use std::error::Error;
use std::time::Instant;

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use kornia::image::allocator::CpuAllocator;
use kornia::image::{Image, ImageSize};
use kornia::imgproc;
use kornia::imgproc::interpolation::InterpolationMode;

type AnyError = Box<dyn Error + Send + Sync + 'static>;
type GpuClient = ComputeClient<WgpuRuntime>;

struct Args {
    iters: usize,
    width: usize,
    height: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut iters = 200usize;
    let mut width = 1920usize;
    let mut height = 1080usize;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    iters = v;
                }
            }
            "--width" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    width = v;
                }
            }
            "--height" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    height = v;
                }
            }
            _ => {}
        }
        i += 1;
    }

    Args {
        iters,
        width,
        height,
    }
}

#[cube(launch_unchecked)]
fn gray_from_rgb_kernel(rgb: &Array<f32>, gray: &mut Array<f32>) {
    if ABSOLUTE_POS < gray.len() {
        let base = ABSOLUTE_POS * 3;
        let r = rgb[base];
        let g = rgb[base + 1];
        let b = rgb[base + 2];
        gray[ABSOLUTE_POS] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
}

#[cube(launch_unchecked)]
fn resize_nearest_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    #[comptime] src_w: usize,
    #[comptime] src_h: usize,
    #[comptime] dst_w: usize,
    #[comptime] dst_h: usize,
    #[comptime] channels: usize,
) {
    let out_idx = ABSOLUTE_POS;
    if out_idx < dst_w * dst_h {
        let x_out = out_idx % dst_w;
        let y_out = out_idx / dst_w;

        let x_src = (x_out * src_w + dst_w / 2) / dst_w;
        let y_src = (y_out * src_h + dst_h / 2) / dst_h;

        let src_pixel = (y_src * src_w + x_src) * channels;
        let dst_pixel = out_idx * channels;

        let mut c = 0usize;
        while c < channels {
            dst[dst_pixel + c] = src[src_pixel + c];
            c += 1;
        }
    }
}

#[cube(launch_unchecked)]
fn resize_bilinear_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    #[comptime] src_w: usize,
    #[comptime] src_h: usize,
    #[comptime] dst_w: usize,
    #[comptime] dst_h: usize,
    #[comptime] channels: usize,
) {
    let out_idx = ABSOLUTE_POS;
    if out_idx < dst_w * dst_h {
        let x_out = out_idx % dst_w;
        let y_out = out_idx / dst_w;

        let scale_x = (src_w - 1) as f32 / dst_w as f32;
        let scale_y = (src_h - 1) as f32 / dst_h as f32;
        let x_src_f = x_out as f32 * scale_x;
        let y_src_f = y_out as f32 * scale_y;

        let x0 = x_src_f as usize;
        let y0 = y_src_f as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let wx = x_src_f - x0 as f32;
        let wy = y_src_f - y0 as f32;

        let dst_pixel = out_idx * channels;

        let mut c = 0usize;
        while c < channels {
            let p00 = src[(y0 * src_w + x0) * channels + c];
            let p01 = src[(y0 * src_w + x1) * channels + c];
            let p10 = src[(y1 * src_w + x0) * channels + c];
            let p11 = src[(y1 * src_w + x1) * channels + c];

            dst[dst_pixel + c] = p00 * (1.0 - wx) * (1.0 - wy)
                + p01 * wx * (1.0 - wy)
                + p10 * (1.0 - wx) * wy
                + p11 * wx * wy;
            c += 1;
        }
    }
}

fn launch_shape_1d(n: usize) -> (CubeCount, CubeDim) {
    let cube_dim_x = 256u32;
    let cubes_x = (n as u32).div_ceil(cube_dim_x);
    (CubeCount::new_1d(cubes_x), CubeDim::new_1d(cube_dim_x))
}

fn sync_client(client: &GpuClient) -> Result<(), AnyError> {
    pollster::block_on(client.sync())?;
    Ok(())
}

fn cpu_gray_from_rgb(
    rgb: &Image<f32, 3, CpuAllocator>,
) -> Result<Image<f32, 1, CpuAllocator>, AnyError> {
    let mut gray = Image::<f32, 1, _>::from_size_val(rgb.size(), 0.0, CpuAllocator)?;
    imgproc::color::gray_from_rgb(rgb, &mut gray)?;
    Ok(gray)
}

fn cpu_resize_nearest(
    src: &Image<f32, 1, CpuAllocator>,
    dst_w: usize,
    dst_h: usize,
) -> Result<Image<f32, 1, CpuAllocator>, AnyError> {
    let mut dst = Image::<f32, 1, _>::from_size_val(
        ImageSize {
            width: dst_w,
            height: dst_h,
        },
        0.0,
        CpuAllocator,
    )?;
    imgproc::resize::resize_native(src, &mut dst, InterpolationMode::Nearest)?;
    Ok(dst)
}

fn cpu_resize_bilinear(
    src: &Image<f32, 1, CpuAllocator>,
    dst_w: usize,
    dst_h: usize,
) -> Result<Image<f32, 1, CpuAllocator>, AnyError> {
    let mut dst = Image::<f32, 1, _>::from_size_val(
        ImageSize {
            width: dst_w,
            height: dst_h,
        },
        0.0,
        CpuAllocator,
    )?;
    imgproc::resize::resize_native(src, &mut dst, InterpolationMode::Bilinear)?;
    Ok(dst)
}

struct BenchResult {
    name: String,
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl BenchResult {
    fn fps(&self) -> f64 {
        1000.0 / self.avg_ms
    }
}

fn bench<F>(name: &str, iters: usize, mut f: F) -> Result<BenchResult, AnyError>
where
    F: FnMut() -> Result<(), AnyError>,
{
    for _ in 0..5 {
        f()?;
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Ok(BenchResult {
        name: name.to_string(),
        avg_ms: avg,
        min_ms: min,
        max_ms: max,
    })
}

fn bench_gpu_kernel<F>(
    name: &str,
    iters: usize,
    client: &GpuClient,
    mut launch: F,
) -> Result<BenchResult, AnyError>
where
    F: FnMut() -> Result<(), AnyError>,
{
    for _ in 0..5 {
        launch()?;
        sync_client(client)?;
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        launch()?;
        sync_client(client)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Ok(BenchResult {
        name: name.to_string(),
        avg_ms: avg,
        min_ms: min,
        max_ms: max,
    })
}

fn bench_gpu_roundtrip<F>(
    name: &str,
    iters: usize,
    client: &GpuClient,
    mut run_once: F,
) -> Result<BenchResult, AnyError>
where
    F: FnMut() -> Result<(), AnyError>,
{
    for _ in 0..5 {
        run_once()?;
        sync_client(client)?;
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        run_once()?;
        sync_client(client)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Ok(BenchResult {
        name: name.to_string(),
        avg_ms: avg,
        min_ms: min,
        max_ms: max,
    })
}

fn print_result(r: &BenchResult) {
    println!(
        "  {:<44} avg: {:>7.2}ms  min: {:>6.2}ms  max: {:>6.2}ms  fps: {:>6.1}",
        r.name,
        r.avg_ms,
        r.min_ms,
        r.max_ms,
        r.fps()
    );
}

fn print_speedup(cpu: &BenchResult, gpu: &BenchResult) {
    let speedup = cpu.avg_ms / gpu.avg_ms;
    println!("  Speedup ({} vs {}): {:.1}x", gpu.name, cpu.name, speedup);
}

fn main() -> Result<(), AnyError> {
    let args = parse_args();
    let w = args.width;
    let h = args.height;
    let iters = args.iters;
    let dst_w = 640usize;
    let dst_h = 640usize;

    println!();
    println!("kornia-gpu-benchmark");
    println!("====================");
    println!("Input  : {}x{} RGB f32", w, h);
    println!("Output : {}x{} grayscale f32 (resize target)", dst_w, dst_h);
    println!("Iters  : {}", iters);
    println!();

    println!("Initializing wgpu (CubeCL)...");
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    println!("GPU ready.");
    println!();

    let n_rgb = w * h * 3;
    let n_gray = w * h;
    let n_out = dst_w * dst_h;
    let rgb_data: Vec<f32> = (0..n_rgb).map(|i| (i % 255) as f32 / 255.0).collect();

    let rgb_image = Image::<f32, 3, _>::new(
        ImageSize {
            width: w,
            height: h,
        },
        rgb_data.clone(),
        CpuAllocator,
    )?;

    println!("[ gray_from_rgb  {}x{} RGB -> grayscale ]", w, h);
    let cpu_gray = bench("CPU gray_from_rgb (kornia-imgproc)", iters, || {
        let _ = cpu_gray_from_rgb(&rgb_image)?;
        Ok(())
    })?;
    print_result(&cpu_gray);

    let rgb_handle = client.create_from_slice(f32::as_bytes(&rgb_data));
    let gray_handle = client.empty(n_gray * std::mem::size_of::<f32>());
    let (gray_cube_count, gray_cube_dim) = launch_shape_1d(n_gray);

    let gpu_gray = bench_gpu_kernel(
        "GPU gray_from_rgb (CubeCL wgpu, kernel-only)",
        iters,
        &client,
        || {
            unsafe {
                gray_from_rgb_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    gray_cube_count.clone(),
                    gray_cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&rgb_handle, rgb_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&gray_handle, n_gray, 1),
                )?;
            }
            Ok(())
        },
    )?;
    print_result(&gpu_gray);
    let _gray_out_once = f32::from_bytes(&client.read_one(gray_handle)).to_vec();
    print_speedup(&cpu_gray, &gpu_gray);

    let gpu_gray_roundtrip = bench_gpu_roundtrip(
        "GPU gray_from_rgb (CubeCL wgpu, upload+kernel+download)",
        iters,
        &client,
        || {
            let rgb_handle = client.create_from_slice(f32::as_bytes(&rgb_data));
            let gray_handle = client.empty(n_gray * std::mem::size_of::<f32>());
            unsafe {
                gray_from_rgb_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    gray_cube_count.clone(),
                    gray_cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&rgb_handle, rgb_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&gray_handle, n_gray, 1),
                )?;
            }
            let _ = f32::from_bytes(&client.read_one(gray_handle)).to_vec();
            Ok(())
        },
    )?;
    print_result(&gpu_gray_roundtrip);
    println!();

    println!(
        "[ resize nearest  {}x{} -> {}x{} grayscale ]",
        w, h, dst_w, dst_h
    );
    let gray_image = cpu_gray_from_rgb(&rgb_image)?;
    let gray_data: Vec<f32> = gray_image.as_slice().to_vec();

    let cpu_nn = bench("CPU resize_nearest (kornia-imgproc)", iters, || {
        let _ = cpu_resize_nearest(&gray_image, dst_w, dst_h)?;
        Ok(())
    })?;
    print_result(&cpu_nn);

    let gray_src_handle = client.create_from_slice(f32::as_bytes(&gray_data));
    let nn_dst_handle = client.empty(n_out * std::mem::size_of::<f32>());
    let (resize_cube_count, resize_cube_dim) = launch_shape_1d(n_out);

    let gpu_nn = bench_gpu_kernel(
        "GPU resize_nearest (CubeCL wgpu, kernel-only)",
        iters,
        &client,
        || {
            unsafe {
                resize_nearest_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    resize_cube_count.clone(),
                    resize_cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&gray_src_handle, gray_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&nn_dst_handle, n_out, 1),
                    w,
                    h,
                    dst_w,
                    dst_h,
                    1,
                )?;
            }
            Ok(())
        },
    )?;
    print_result(&gpu_nn);
    let _nn_out_once = f32::from_bytes(&client.read_one(nn_dst_handle)).to_vec();
    print_speedup(&cpu_nn, &gpu_nn);

    let gpu_nn_roundtrip = bench_gpu_roundtrip(
        "GPU resize_nearest (CubeCL wgpu, upload+kernel+download)",
        iters,
        &client,
        || {
            let gray_src_handle = client.create_from_slice(f32::as_bytes(&gray_data));
            let nn_dst_handle = client.empty(n_out * std::mem::size_of::<f32>());
            unsafe {
                resize_nearest_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    resize_cube_count.clone(),
                    resize_cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&gray_src_handle, gray_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&nn_dst_handle, n_out, 1),
                    w,
                    h,
                    dst_w,
                    dst_h,
                    1,
                )?;
            }
            let _ = f32::from_bytes(&client.read_one(nn_dst_handle)).to_vec();
            Ok(())
        },
    )?;
    print_result(&gpu_nn_roundtrip);
    println!();

    println!(
        "[ resize bilinear {}x{} -> {}x{} grayscale ]",
        w, h, dst_w, dst_h
    );

    let cpu_bl = bench("CPU resize_bilinear (kornia-imgproc)", iters, || {
        let _ = cpu_resize_bilinear(&gray_image, dst_w, dst_h)?;
        Ok(())
    })?;
    print_result(&cpu_bl);

    let bl_dst_handle = client.empty(n_out * std::mem::size_of::<f32>());

    let gpu_bl = bench_gpu_kernel(
        "GPU resize_bilinear (CubeCL wgpu, kernel-only)",
        iters,
        &client,
        || {
            unsafe {
                resize_bilinear_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    resize_cube_count.clone(),
                    resize_cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&gray_src_handle, gray_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&bl_dst_handle, n_out, 1),
                    w,
                    h,
                    dst_w,
                    dst_h,
                    1,
                )?;
            }
            Ok(())
        },
    )?;
    print_result(&gpu_bl);
    let _bl_out_once = f32::from_bytes(&client.read_one(bl_dst_handle)).to_vec();
    print_speedup(&cpu_bl, &gpu_bl);

    let gpu_bl_roundtrip = bench_gpu_roundtrip(
        "GPU resize_bilinear (CubeCL wgpu, upload+kernel+download)",
        iters,
        &client,
        || {
            let gray_src_handle = client.create_from_slice(f32::as_bytes(&gray_data));
            let bl_dst_handle = client.empty(n_out * std::mem::size_of::<f32>());
            unsafe {
                resize_bilinear_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    resize_cube_count.clone(),
                    resize_cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&gray_src_handle, gray_data.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&bl_dst_handle, n_out, 1),
                    w,
                    h,
                    dst_w,
                    dst_h,
                    1,
                )?;
            }
            let _ = f32::from_bytes(&client.read_one(bl_dst_handle)).to_vec();
            Ok(())
        },
    )?;
    print_result(&gpu_bl_roundtrip);
    println!();

    println!("=======================================================");
    println!(
        " SUMMARY  ({}x{} -> {}x{}, {} iters)",
        w, h, dst_w, dst_h, iters
    );
    println!("=======================================================");
    println!(
        " gray_from_rgb   CPU {:>7.2}ms  GPU {:>6.2}ms  {:>5.1}x speedup",
        cpu_gray.avg_ms,
        gpu_gray.avg_ms,
        cpu_gray.avg_ms / gpu_gray.avg_ms
    );
    println!(
        " resize_nearest  CPU {:>7.2}ms  GPU {:>6.2}ms  {:>5.1}x speedup",
        cpu_nn.avg_ms,
        gpu_nn.avg_ms,
        cpu_nn.avg_ms / gpu_nn.avg_ms
    );
    println!(
        " resize_bilinear CPU {:>7.2}ms  GPU {:>6.2}ms  {:>5.1}x speedup",
        cpu_bl.avg_ms,
        gpu_bl.avg_ms,
        cpu_bl.avg_ms / gpu_bl.avg_ms
    );
    println!("=======================================================");
    println!(" Round-trip timings (upload + kernel + download):");
    println!(
        " gray_from_rgb   CPU {:>7.2}ms  GPU {:>6.2}ms  {:>5.1}x speedup",
        cpu_gray.avg_ms,
        gpu_gray_roundtrip.avg_ms,
        cpu_gray.avg_ms / gpu_gray_roundtrip.avg_ms
    );
    println!(
        " resize_nearest  CPU {:>7.2}ms  GPU {:>6.2}ms  {:>5.1}x speedup",
        cpu_nn.avg_ms,
        gpu_nn_roundtrip.avg_ms,
        cpu_nn.avg_ms / gpu_nn_roundtrip.avg_ms
    );
    println!(
        " resize_bilinear CPU {:>7.2}ms  GPU {:>6.2}ms  {:>5.1}x speedup",
        cpu_bl.avg_ms,
        gpu_bl_roundtrip.avg_ms,
        cpu_bl.avg_ms / gpu_bl_roundtrip.avg_ms
    );
    println!("=======================================================");
    println!("Note: kernel-only timings reuse persistent GPU buffers.");
    println!("Note: round-trip timings include upload, kernel launch, sync, and download.");

    Ok(())
}
