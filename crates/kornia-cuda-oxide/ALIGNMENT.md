# kornia-cuda-oxide — alignment with project proposal

This branch (`proto/cuda-oxide`) is the **cuda-oxide counterpart** to the existing
`proto/cubecl` prototype (PR #897). Both are sub-workspace prototypes whose
purpose is to give the broader kornia-rs GPU initiative empirical data to make
architectural decisions on.

The companion `proto/cubecl` prototype lives at `crates/kornia-cubecl/` on its
own branch and is already merged-ready as a draft PR.

## Scope of this branch

**This branch delivers Goal 5 (backend selection) and Goal 6 (perf benchmarking)
from the project proposal. It explicitly defers Goals 1-4 to a follow-on
workstream.**

Rationale: Goal 5 reads *"The selected backend will be locked early to avoid
mid-project architectural changes."* That makes it a prerequisite gate — you
cannot sensibly design `GpuAllocator<Backend>`'s associated-type machinery, or
backend-specialized tensor op traits, until the backend is chosen. Bind cubecl's
`ComputeClient<R>` into the allocator type and you're locked into cubecl
forever; bind cuda-oxide's `CudaContext + DeviceBuffer` and you're locked into
that. The architectural work in Goals 1-4 only makes sense *after* this
decision.

## Goal-by-goal mapping

| # | Proposal goal | Status on this branch | Where it actually lands |
|---|---|---|---|
| 1 | `GpuAllocator` in kornia-tensor | **Deferred** | Follow-on branch off main, post-backend-selection |
| 2 | Backend-specialized tensor op traits | **Deferred** | Same follow-on |
| 3 | GPU execution for kornia-imgproc ops | **Deferred** | Same follow-on |
| 4 | API compatibility / explicit memory semantics | Trivially satisfied | Sub-workspace prototype crate — does not touch user-facing APIs |
| 5 | **Backend selection (cubecl vs CUDA bindings)** | **Delivered** | Kernel port + perf/ergonomics scorecard in `COMPARISON.md` |
| 6 | **Perf benchmarking, H↔D separated from kernel** | **Delivered** | `examples/bench_min.rs` + `examples/bench_fusion.rs` (mirroring kornia-cubecl) |

## What this branch produces

Mirrors the structure of `crates/kornia-cubecl/`:

- **Same kernel set:** bilinear resize (baseline + tiled variants + pre-uploaded weights), RGB→gray, normalize, fused resize+gray+norm, fused resize+CHW+norm
- **Same bench harness:** identical sizes, identical methodology (10 reps, 3 warmup, median), kernel-only vs end-to-end arms separated
- **Same correctness gate:** bit-exact vs `fast_image_resize` reference
- **New deliverable:** `COMPARISON.md` — head-to-head cubecl vs cuda-oxide numbers + qualitative scorecard on:
  - Integration complexity (toolchain, build, packaging)
  - Safety model (`unsafe` surface area, kernel-launch invariants)
  - Cross-platform considerations (cubecl: CPU+CUDA+ROCm+WGPU; cuda-oxide: CUDA-only)
  - Kernel development ergonomics (macro DSL vs single-source Rust)

## Why the deferred goals are deferred, not abandoned

The follow-on branch will need:

- A `GpuAllocator<B: Backend>` trait whose associated types name the chosen
  backend's runtime types (whichever wins the bake-off)
- Backend-specialized impls (e.g. `impl<B> Resize for Tensor<u8, 3, GpuAllocator<B>>`)
- An MVP tensor op set (add, mul, abs, relu, clamp, sum, mean) that the cubecl
  prototype doesn't have — these are new kernels regardless of backend choice
- Imgproc dispatch for resize / warp_affine / warp_perspective / RGB↔gray

None of those should be built against both backends in parallel; once Goal 5
locks, they are built once against the winner. That's the explicit point of
locking the backend early.

## Notes on the proposal's design constraints

A few constraints from the proposal that are worth surfacing because they
affect backend evaluation:

- **"No arbitrary closures on GPU"** (Goal 2) — predefined op enums / trait-specialized
  kernels only. This declines cuda-oxide's headline closure-capture feature, but
  cubecl's `#[cube]` doesn't accept closures either, so the constraint favors
  neither backend. Both must expose only declared, deterministic kernels.
- **"No measurable CPU regression"** (Goal 6) — the sub-workspace pattern this
  branch uses sidesteps this trivially (main workspace resolver untouched). The
  follow-on branch will need real CPU benches once we start integrating into
  `kornia-tensor`.
