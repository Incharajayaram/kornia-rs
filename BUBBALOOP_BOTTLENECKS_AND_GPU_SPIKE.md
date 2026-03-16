# Bubbaloop Workload Analysis -> Kornia-rs GPU Deliverables

This document reframes the analysis correctly for project scope:

- `bubbaloop` is used as a real workload reference.
- Implementation work remains in `kornia-rs` (`kornia-tensor` and `kornia-imgproc`).
- No bubbaloop-internal fixes are part of this project unless explicitly requested later.

## 1) Why bubbaloop was inspected

I inspected bubbaloop to extract concrete memory and latency patterns that justify the GPU work in `kornia-rs`:

1. High-rate camera frames (`CompressedImage`, H264 byte payloads).
2. Strict low-latency stream behavior (`BestEffort + Drop`).
3. Cross-machine operation where host/device copies dominate if not controlled.
4. Real deployment shape: camera input -> preprocess -> inference/decision.

So the proposal is not "improve bubbaloop internals"; the proposal is "implement kornia-rs GPU primitives that solve the bottlenecks exposed by bubbaloop workloads."

## 2) Observed bottlenecks and how they map to kornia-rs work

### 2.1 Memory domain ambiguity leads to hidden copy risk

Observed in workload:
- Camera pipelines are copy-sensitive and latency-sensitive.
- End-to-end "zero-copy" is not guaranteed across all boundaries (local SHM is not equivalent to full pipeline zero-copy).

Proposal workaround in kornia-rs:
- Enforce explicit memory domains in tensor storage (`Host` vs `Device`).
- Remove accidental host access for device tensors (compile-time host API gating).
- Keep transfers explicit (`to_device`, `to_host`) with no implicit fallback copy.

Concrete kornia-rs deliverables:
1. Feature-gated `GpuAllocator` implementing existing allocator trait.
2. Domain-aware storage model in `kornia-tensor`.
3. Compile-time restriction for host-only APIs (`as_slice`, `into_vec`) on device tensors.
4. Explicit transfer API with tests for ownership and drop correctness.

### 2.2 Per-frame compute path needs output-parallel preprocessing

Observed in workload:
- Camera-derived operations are per-frame and per-pixel, and latency accumulates fast at 1080p/4K.

Proposal workaround in kornia-rs:
- Move selected imgproc operations to allocator-specialized GPU kernels.
- Use output-parallel mapping (one thread per output pixel) for resize/warp/color paths.

Concrete kornia-rs deliverables:
1. Allocator-specialized dispatch traits for imgproc execution.
2. GPU kernels for:
   - `resize` (nearest, bilinear)
   - `warp_affine`
   - `warp_perspective`
   - `RGB <-> Grayscale`
3. CPU/GPU parity enforcement by preserving CPU math semantics (inverse mapping + interpolation consistency).

### 2.3 Tensor-level primitive cost needs backend-specialized execution

Observed in workload:
- Repeated elementwise/reduction style transforms in preprocessing stages become throughput bottlenecks when performed on host memory.

Proposal workaround in kornia-rs:
- Introduce allocator-specialized tensor op traits and GPU kernels for predefined operation sets.
- Avoid arbitrary closure ABI on GPU; use deterministic operation enums.

Concrete kornia-rs deliverables:
1. GPU-backed tensor unary/binary/reduction dispatch.
2. Operation sets:
   - Unary: `Abs`, `Relu`, `Clamp`, `Neg`
   - Binary: `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`
   - Reduce: `Sum`, `Mean`
3. Structured unsupported-path errors (no silent fallback unless explicitly requested).

### 2.4 Real-time behavior requires benchmark split by transfer vs compute

Observed in workload:
- In stream pipelines, total latency is dominated by both transfer and kernel time; only reporting kernel time is insufficient.

Proposal workaround in kornia-rs:
- Benchmarks will report `H2D`, kernel, and `D2H` separately.
- Validate speedup only under realistic image sizes and shapes used in pipeline-like workloads.

Concrete kornia-rs deliverables:
1. Layered benchmarks:
   - Layer A: micro kernels
   - Layer B: tensor ops
   - Layer C: imgproc (1080p, 4K, batch)
2. Acceptance metrics:
   - numerical parity tolerance vs CPU
   - speedup target on defined resolutions
   - transfer-accounted latency reporting
3. Regression checks in benchmark CI path (feature-gated GPU jobs).

## 3) Backend spike summary (de-risking implementation path)

These tests were run in isolated temporary crates (`/tmp/cubecl_smoke`, `/tmp/wgpu_smoke`), not inside bubbaloop.

### 3.1 CubeCL viability

1. `cubecl` + `wgpu` compile: pass.
2. `cubecl` + `cuda` compile: pass.
3. `cubecl` + `wgpu` runtime smoke (client init + create/read): pass.
4. Quick `wgpu` transfer micro-smoke (`create_from_slice` + `read_one` loop): pass.

### 3.2 CUDA runtime note

1. `cubecl` + `cuda` runtime smoke failed on this machine due to a CUDA driver/runtime symbol mismatch (`cuDevSmResourceSplit` in `libcuda.so`).
2. This does not block proposal scope, but it supports keeping CUDA as fallback with explicit environment constraints.

### 3.3 Direct raw wgpu viability

1. Native `wgpu` adapter/device initialization smoke passed (`Vulkan`, NVIDIA GTX 1650 detected).
2. This confirms that `wgpu` execution path is practical for first-stage bring-up.

### 3.4 Raw spike metrics (added for backend justification)

These are not full kernel benchmarks. They are transfer/init micro-spikes used to compare bring-up viability and baseline overhead.

Environment snapshot:
- GPU: NVIDIA GeForce GTX 1650
- Backend detected by `wgpu`: Vulkan
- Runtime in CubeCL run: `wgpu<wgsl>`
- Sample sizes: 1 MB, 8 MB, 32 MB
- Iterations per size: 25
- Measurement style: average and p50 latency in milliseconds

Raw `wgpu` spike (second run after build warmup):

| Metric | Value |
|---|---:|
| Adapter init (ms) | 261.460 |
| Device init (ms) | 129.289 |

| Size | H2D avg (ms) | H2D p50 (ms) | D2H avg (ms) | D2H p50 (ms) |
|---|---:|---:|---:|---:|
| 1 MB | 0.608 | 0.548 | 0.498 | 0.467 |
| 8 MB | 3.529 | 3.462 | 3.047 | 2.940 |
| 32 MB | 13.019 | 12.871 | 12.092 | 11.500 |

CubeCL on `wgpu` runtime spike (second run after build warmup):

| Metric | Value |
|---|---:|
| Client init (ms) | 255.757 |

| Size | H2D avg (ms) | H2D p50 (ms) | D2H avg (ms) | D2H p50 (ms) |
|---|---:|---:|---:|---:|
| 1 MB | 0.366 | 0.164 | 1.180 | 1.030 |
| 8 MB | 2.380 | 1.616 | 9.697 | 9.309 |
| 32 MB | 29.311 | 29.080 | 34.869 | 33.938 |

Interpretation for proposal:
1. Both paths are operational on this machine (raw `wgpu` and CubeCL `wgpu` runtime).
2. CubeCL client bring-up cost is in the same order as raw `wgpu` device init.
3. The CubeCL micro includes per-iteration resource creation/readback overhead, so it should not be treated as final kernel-performance evidence.
4. This supports using CubeCL + `wgpu` for early integration while keeping benchmark-gated backend decisions for later phases.

What this data is and is not:
- Useful for: backend bring-up risk reduction and early runtime choice rationale.
- Not sufficient for: final backend selection on compute performance.
- Final selection still requires op-level benchmarks (Layer A/B/C) with explicit transfer split and real kernels.

## 4) How this is reflected in the current proposal

The current proposal can state this explicitly:

1. Primary implementation target:
   - CubeCL backend with `wgpu` runtime path for early stability and portability.
2. Fallback path:
   - CUDA (`cust` or CubeCL CUDA runtime) behind feature flag, activated only if defined performance/correctness criteria require it.
3. Use-case grounding:
   - Bubbaloop camera pipeline is the motivating workload used to define operation priority and benchmark scenarios.
4. Scope boundary:
   - Deliverables are exclusively in `kornia-rs`; bubbaloop code is used only as external workload reference.

## 5) Final kornia-rs deliverable list from this workload mapping

1. `kornia-tensor`:
   - `GpuAllocator`
   - domain-aware storage
   - compile-time host API gating
   - explicit transfer APIs
2. `kornia-tensor-ops`:
   - allocator-specialized unary/binary/reduction dispatch
   - GPU kernels for predefined ops
3. `kornia-imgproc`:
   - GPU kernels for resize/warp/color
   - allocator-agnostic public API with backend dispatch
4. Validation:
   - parity tests, boundary tests, compile-fail domain tests
   - transfer-accounted benchmarks at 1080p/4K + batching
5. Documentation:
   - GPU backend usage docs
   - examples of explicit host/device workflow
   - compatibility/status tables for op/dtype/backend coverage
