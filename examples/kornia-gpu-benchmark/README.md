# Kornia GPU Benchmark Demo

This example crate was used to benchmark the `kornia-rs` GPU preprocessing path against the CPU path.

It covers:

- `gray_from_rgb`
- `resize_nearest`
- `resize_bilinear`

The GPU path uses `cubecl 0.9.0` with the `wgpu` backend.

## What This Demo Shows

This benchmark is useful for two parts of the overall demo:

1. Local GPU image preprocessing benchmarks on a laptop GPU.
2. A reproducible baseline for the preprocessing stage before wiring the same ideas into a Bubbaloop node.

For the VLM portion of the demo, use the `kornia-vlm` test commands further below instead of this crate.

## Prerequisites

- Rust toolchain
- A working Vulkan/WGPU-capable GPU
- Release builds only

Optional local dependency install on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev
```

## Build

```bash
cd ~/kornia-rs/examples/kornia-gpu-benchmark
cargo build --release
```

## Run

Default run:

```bash
cd ~/kornia-rs/examples/kornia-gpu-benchmark
cargo run --release
```

More stable averages:

```bash
cd ~/kornia-rs/examples/kornia-gpu-benchmark
cargo run --release -- --iters 200
```

4K benchmark:

```bash
cd ~/kornia-rs/examples/kornia-gpu-benchmark
cargo run --release -- --iters 200 --width 3840 --height 2160
```

## Notes

- Use `--release`. Debug mode gives misleading CPU timings.
- These numbers are kernel-focused and do not represent full application pipeline timings.
- Full pipeline timings were measured in the Bubbaloop node demo.

## Cloud VLM Validation

The VLM part of the demo was validated separately on a cloud GPU.

The setup that worked was:

- GPU: `NVIDIA L40S`
- CUDA toolkit: `12.5`
- build features: `cuda flash-attn`

### Required environment on the cloud GPU

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.5/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:${LD_LIBRARY_PATH}
export CUDA_ROOT=/usr/local/cuda-12.5
export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.5
export CUDARC_CUDA_VERSION=12050
export CUDA_COMPUTE_CAP=89
```

### SmolVLM2 sanity test

```bash
cd ~/kornia-rs
cargo test --release -p kornia-vlm test_smolvlm2_text_inference --features "cuda flash-attn" -- --nocapture --ignored
```

### SmolVLM2 image inference benchmark

```bash
cd ~/kornia-rs
cargo test --release -p kornia-vlm test_smolvlm2_image_inference_speed --features "cuda flash-attn" -- --nocapture --ignored
```

This prints:

- model initialization time
- average image inference time
- min/max image inference time
- token generation throughput in the debug output

Reference result from the successful validation run:

- `SmolVLM2` image inference on `L40S`
- roughly `157-159 tokens/s`
- total test runtime `4.82s`

## Related Demo

The corresponding Bubbaloop node demo lives in:

- `../../bubbaloop/kornia-gpu-node`

That node was used to measure end-to-end webcam pipeline timings and to publish the processed stream to the dashboard.
