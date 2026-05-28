# cubecl vs cuda-oxide — Goal-5 backend bake-off

**Status:** Updated 2026-05-28 — helper-call ICE confirmed fixed upstream
(NVlabs/cuda-oxide#79, commit `6ed9938`); `with_helper_kernel` reproducer now
compiles cleanly against `origin/main` (see "Additional architectural gotcha"
section below for verification details). Backend choice unchanged: cubecl
remains the selection for Coding Phase 1 per mentor steer (alpha churn,
toolchain friction, library-propagation gap still on cuda-oxide's side); the
ICE class is no longer a blocker but the broader maturity gap is.

Previous update (2026-05-24) added complete tiled-kernel optimization
investigation (seq-macro unrolling, LLVM unroll flags, get_unchecked) and
confirmed root cause of x4/x16 underperformance. Baseline resize, fused
gray+normalize (P1), and fused CHW-normalize ML preprocessing (P2) numbers
unchanged.

**Hardware:** Single host, head-to-head on the same physical GPU.
- CPU: x86_64 (Tiger Lake), Ubuntu 22.04
- GPU: NVIDIA GTX 1650 (Turing, sm_75, 4 GB GDDR6, ~128 GB/s, PCIe Gen3 x16)
- CUDA toolkit: 12.4 (nvcc 12.4.131)

**Methodology:**
- Same input sizes (6: 512² → 8K + 1080p→540p), same seed (`0xC0FFEE`), same
  10-rep / 3-warmup median methodology on both backends.
- Bench arms separated by transfer-tax visibility (Goal 6 requirement):
  - `kernel`: pre-uploaded buffers + pre-loaded module, times only kernel
    dispatch + `stream.synchronize()`.
  - `e2e`: fresh device-buffer allocation + upload + launch + download per rep.
- Both prototypes compile the same Rust source (fixed-point bilinear,
  identical weight-table layout) to PTX; cubecl via JIT/MLIR, cuda-oxide via
  AOT LLVM-21 nvptx.

## Correctness gate

| backend | sizes | result |
|---|---|---|
| **cubecl** (proto/cubecl, PR #897) | 512², 1024², 2048², 4096² | bit-exact vs `fast_image_resize`, max_diff = 0 |
| **cuda-oxide** (this branch) | 512², 1024², 2048², 4096² | bit-exact vs `fast_image_resize`, max_diff = 0 |

Both backends produce **identical bytes** to the production CPU reference.
The cuda-oxide kernel is a near-line-for-line port of the cubecl kernel —
same fixed-point arithmetic, same `(1 << 15) + >> 16` rounding policy. The
correctness equivalence means we can compare them purely on perf + ergonomics
without worrying about output drift.

## Resize-only throughput (kernel-only, median μs / Mpix/s)

Apples-to-apples: both pre-upload weights once and reuse across reps. For
cubecl this is the `_pw` (pre-uploaded weights) variant; for cuda-oxide the
weight buffers live outside the timing loop.

| src → dst         | cubecl `_pw` (μs) | cubecl `_pw` (Mpix/s) | **cuda-oxide** (μs) | **cuda-oxide** (Mpix/s) | cuda-oxide vs cubecl |
|-------------------|------------------:|----------------------:|--------------------:|------------------------:|---------------------:|
| 512² → 256²       |              35.7 |                   919 |            **14.1** |                **2322** |    **2.53× faster** |
| 1024² → 512²      |              54.2 |                  2418 |            **31.5** |                **4157** |    **1.72× faster** |
| 2048² → 1024²     |             105.6 |                  4966 |            **95.9** |                **5466** |        1.10× faster |
| 4096² → 2048²     |             345.4 |                  6071 |               354.6 |                    5915 |   1.03× *slower* (tied) |
| 8192² → 4096²     |             1471  |                  5700 |          **1216**   |                **6900** |    **1.21× faster** |
| 1920×1080 → 960×540 |           101.3 |                  5116 |            **79.8** |                **6495** |    **1.27× faster** |

For reference vs **raw cubecl** (no weights pre-upload, reflecting a
"naive consumer" of the cubecl API):

| size | cubecl baseline (μs) | cuda-oxide (μs) | cuda-oxide vs cubecl baseline |
|---|---:|---:|---:|
| 512²  |  87.4 |  14.1 | **6.2× faster** |
| 1024² |  88.6 |  31.5 | **2.81× faster** |
| 2048² | 204.2 |  95.9 | **2.13× faster** |
| 4096² | 455.9 | 354.6 | 1.29× faster |
| 8192² | 1691  | 1216  | 1.39× faster |
| 1080p | 171.0 |  79.8 | **2.14× faster** |

**Reading these:** cuda-oxide's AOT LLVM-21 nvptx codegen produces a kernel
that performs comparably to (and at small sizes, materially better than)
cubecl's JIT-compiled MLIR-cuda kernel. The 2-2.5× wins at small sizes reflect
both (a) leaner kernel code emitted by LLVM-21 and (b) lower per-launch
dispatch overhead in cuda-oxide's thinner host runtime. At ≥2048² output, both
backends are bandwidth-bound and converge — within ±3% on the same GPU.

## End-to-end throughput (includes H↔D copy, median μs / Mpix/s)

| src → dst | cubecl e2e (μs) | cubecl e2e (Mpix/s) | **cuda-oxide e2e** (μs) | **cuda-oxide e2e** (Mpix/s) | cuda-oxide vs cubecl |
|---|---:|---:|---:|---:|---:|
| 512²  |      788 |  41.6 |        **394** | **83**  | **2.0× faster** |
| 1024² |     3408 |  38.5 |        **814** | **161** | **4.2× faster** |
| 2048² |    15267 |  34.3 |       **3315** | **158** | **4.6× faster** |
| 4096² |    41284 |  50.8 |      **13230** | **158** | **3.1× faster** |
| 8192² |   383579 |  21.9 |      **55040** | **152** | **7.0× faster** |
| 1080p |    11510 |  45.0 |       **3605** | **144** | **3.2× faster** |

The 3-7× e2e gap is much larger than the kernel-only gap. Likely
explanation: cubecl's `create_from_slice` path on discrete CUDA (over PCIe)
does redundant intermediate copies / staging that cuda-oxide's
`DeviceBuffer::from_host` avoids. The PR #897 notes already flagged
"end-to-end cuda is dominated by cudaMemcpy" as a known cubecl limitation on
non-Tegra hardware; cuda-oxide doesn't appear to share it.

This gap is consequential for kornia-rs: any production pipeline that does
per-frame H↔D round-trips (which is the common pattern for ML preprocessing
streaming) will see a 3-7× e2e win from cuda-oxide on discrete GPUs.

## Qualitative scorecard

The proposal's Goal-5 selection criteria are integration complexity, safety
model, cross-platform considerations, and kernel-development ergonomics. Our
takes after building one kernel end-to-end on both:

### Integration complexity

| | cubecl (proto/cubecl) | cuda-oxide (this branch) |
|---|---|---|
| Build toolchain | stable Rust, normal `cargo build` | nightly-2026-04-03 pinned, custom rustc backend (LLVM-21 nvptx) |
| First-build cost | ~5-15 min (cubecl + cudarc tree) | ~10 min (incl. building rustc-codegen-cuda backend) |
| Workspace impact | Heavy resolver pressure — needed sub-workspace to isolate | Same — sub-workspace pattern still required |
| Dep tree | cubecl 0.10-pre.4 + cudarc + tracel-llvm bundle for cpu | cuda-oxide path deps + LLVM 21 + clang 21 system packages |
| Per-machine setup gotchas | `CUDARC_CUDA_VERSION=<NNNN>` env var to avoid libcuda dlsym mismatch; tracel-llvm bundle dir mislabel | `CUDA_OXIDE_TARGET=sm_<NN>` env var to avoid driver-218 PTX JIT failure; LLVM 21 + clang 21 + libclang-common-21-dev must be installed; nightly toolchain pinned via `rust-toolchain.toml` |
| **Verdict** | both have real per-machine setup friction. cubecl's friction is in the dep-graph (cudarc symbol mapping); cuda-oxide's is in system toolchain (LLVM/clang/nightly). | **comparable; cuda-oxide's is more "real toolchain", cubecl's is more "Rust resolver".** |

### Library-pattern fit

The single biggest discovery for kornia-tensor integration:

| | cubecl | cuda-oxide |
|---|---|---|
| Multi-binary consumers (one lib, many bins/examples) | Just works | **`#[cuda_module]` pattern silently fails — embed.o has no exported symbols, linker strips it for non-bin consumers. Must use free `#[kernel]` + `load_module_from_file` instead.** |
| Embedded PTX | n/a (JIT-compiled at runtime) | yes via `#[cuda_module]`, but only for single-bin crates |
| Separate `.ptx` file | n/a | yes via free `#[kernel]`; works for libs |

This matters for the proposal's follow-on work (GpuAllocator, kornia-imgproc
dispatch). The natural pattern of "lib defines kernel, multiple consumers use
it" requires the free `#[kernel]` + `.ptx` file path in cuda-oxide. This is
fine but **adds a runtime file dependency** (the binary needs to find the
`.ptx` next to where it was built), which the cubecl path doesn't have. For
production kornia distribution, this means either bundling the `.ptx` as a
resource or embedding it via `include_bytes!`.

### Safety model

| | cubecl | cuda-oxide |
|---|---|---|
| Kernel body | safe Rust via `#[cube]` proc macro DSL | safe Rust + standard `#[kernel]` attr |
| Array indexing | `Array<u8>` — bounds-checked at debug; UB in release | `&[u8]` (read) + `DisjointSlice<T>` (write) — `DisjointSlice` enforces unique thread-index access via the type system (`ThreadIndex<IS>` witness); explicit `unsafe` for `get_unchecked_mut` |
| Out-of-bounds threads | `terminate!()` (early return) | `return` after manual `index_2d_*` check |
| Generic kernel codegen | comptime monomorphization | const generics + runtime-stride 2D index escape hatch (unsafe) |
| **Verdict** | cuda-oxide's type-witness-based unique-access proof is more rigorous on paper. cubecl is more permissive. | **cuda-oxide stronger by design;** the proof obligation is right at the index construction site (`unsafe index_2d_runtime`) rather than dispersed through indexing operations. |

### Cross-platform considerations

| backend | CUDA | ROCm | WebGPU | CPU (MLIR) | Apple Metal |
|---|:-:|:-:|:-:|:-:|:-:|
| **cubecl** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **cuda-oxide** | ✓ | ✗ | ✗ | ✗ | ✗ |

cuda-oxide is **CUDA-only by design** (it's literally a rustc backend that
emits PTX). cubecl is multi-backend by design.

This is the single biggest open question for the backend selection. The
proposal mentions "Cross-platform considerations" as Goal-5 criterion, and
cubecl is unambiguously stronger here. Any kornia-rs user on AMD GPU, AMD
laptop, M1/M2 Mac, or "no GPU at all" gets nothing from cuda-oxide. Whereas
cubecl-cpu (MLIR) is a real fallback path, and cubecl-wgpu would work for
browser-deployed pipelines.

### Kernel-development ergonomics

| | cubecl | cuda-oxide |
|---|---|---|
| Syntax | proc-macro DSL (`#[cube]`, `#[comptime]`, `terminate!()`, `Array<T>`) | standard Rust (`#[kernel]`, `&[T]`, `return`, `DisjointSlice<T>`) |
| IDE support | rust-analyzer struggles with `#[cube]` body (DSL token rewriting) | rust-analyzer works normally |
| Debug story | print debugging via prefetch buffer | `cuda_oxide_kernel_<hash>` symbol names + `cuda-gdb` integration |
| Composable primitives | yes (one `#[cube] pub fn` per primitive, used by inlining into a fused `#[cube(launch)]` kernel) | yes (regular Rust fn referenced by the `#[kernel]`) |
| Comptime / compile-time monomorphization | first-class (`#[comptime] src_w: u32`) | const generics; less ergonomic for shape-varying kernels (must use `index_2d_runtime` with `unsafe`) |
| **Verdict** | **cubecl wins on shape specialization,** cuda-oxide wins on "feels like normal Rust" and tooling. | both viable. |

## Architectural gotchas worth documenting

Three things bit us during this port that would bite the eventual
kornia-tensor integration team identically:

1. **`#[cuda_module]` doesn't propagate from libs to binaries.** The embed.o
   it produces has no exported symbols, so the linker strips it. Use free
   `#[kernel]` + `ctx.load_module_from_file("name.ptx")` for any kernel
   defined in a crate that's consumed by multiple binaries.

2. **`CUDA_OXIDE_TARGET=sm_<NN>` is mandatory.** Default PTX target is
   sm_80; running on anything older (sm_75 = Turing including our GTX 1650
   and tons of consumer GPUs) gives a cryptic `DriverError(218, "a PTX JIT
   compilation failed")` at module-load time. No compile-time check.

3. **PTX lives at crate root, not in `target/`.** The codegen backend writes
   `<crate_name>.ptx` next to `src/`. Use `env!("CARGO_MANIFEST_DIR")` at
   the consumer's compile time to construct an absolute path, otherwise
   `cargo run` from a different CWD will fail to find it.

The RUSTFLAGS that `cargo oxide build` sets internally and that we
hand-replicate for `cargo test` / `cargo build --example`:

```
-Z codegen-backend=$BACKEND_SO
-C opt-level=3
-C debug-assertions=off
-Z mir-enable-passes=-JumpThreading
-C symbol-mangling-version=v0
```

## Tiled variants (x1, x4, x16) — kernel-only median μs

Post-optimization state (seq-macro source-level unrolling, see investigation
section below):

| Size | cuda-oxide x1 | cuda-oxide x4 | cuda-oxide x16 | cubecl x1 (ref) | cubecl x16 (ref) |
|---|---:|---:|---:|---:|---:|
| 512² | 14.1 | 27.3 | **48.9** (3.5× slower than x1) | 110 | 105 |
| 1024² | 32.6 | 43.6 | **409** (12.5× slower than x1) | 88 | 226 |
| 2048² | 97.6 | 128.9 | **2108** (21.6× slower than x1) | 204 | 521 |
| 4096² | 300 | 373 | **9170** (30.6× slower than x1) | 456 | 1743 |
| 8K | 1260 | 1496 | **36489** (29× slower than x1) | 1691 | 6040 |
| 1080p | 81.0 | 107.9 | **2126** (26.2× slower than x1) | 171 | 884 |

**All variants are bit-exact correct vs `fast_image_resize` (max_diff = 0).**

### Tiled kernel optimization investigation

We exhaustively tried all planned loop-unrolling approaches and PTX-inspected
each result. Summary of findings:

**Step 1.1 — `for px in 0..N` (range loops):**
Changed `while px < 16 { ... px += 1; }` → `for px in 0..16 { ... }`. LLVM
did not unroll. In fact, `for` loops are *worse* than `while` for
cuda-oxide/pliron because the `Range<usize>` iterator desugars to an
`Option<usize>` state machine that pliron lowers to a complex FSM with 16-bit
flag checks per iteration, adding 3-4 instructions of iterator-protocol
overhead per cycle.

**Step 1.2 — `--unroll-threshold=4096` LLVM flag:**
Added `-C llvm-args=--unroll-threshold=4096` to RUSTFLAGS. No effect.
LLVM-21 nvptx's loop unroller either doesn't fire for NVPTX targets or the
threshold flag isn't being respected by the nvptx backend.

**Step 1.4 — `seq-macro` source-level expansion:**
Added `seq-macro = "0.3"` dependency and rewrote tiled kernels using nested
`seq!(PX in 0..16 { ... })`. This **fully eliminated all loops from the PTX**
(1,581 → 4,511 PTX lines; zero "Loop Header" annotations). x4 improved ~15%;
x16 improved ~4% — but x16 remained 20-29× slower than x1.

**Root cause (identified from PTX inspection):**

1. **226 bounds checks in x16 PTX** — each `src[row_top + off_l + CH]`
   access generates a `setp.ge + @%p bra $exit` pair. With 48 seq-expanded
   iterations × 5 checks each = 226 conditional branches that serialize
   all memory loads (the `ld.b8` can't be scheduled before its bounds check
   resolves). By contrast x1 has only 17 such branches.

2. **Non-coalesced weight array access** — in a 32-thread warp, the x16 kernel
   has threads at tile_x=0,1,...,31 accessing `weights_x_idx[PX]`,
   `weights_x_idx[16+PX]`, `weights_x_idx[32+PX]`, ... at **stride-16**
   (64 bytes between adjacent threads). This requires 32 separate cache lines
   per weight load vs x1's stride-1 (1 cache line for 32 threads). Over 16
   iterations this generates 512× more cache pressure vs x1.

**`get_unchecked` attempt (failed):** Replacing `src[idx]` with
`unsafe { *src.get_unchecked(idx) }` eliminated bounds-check branches but
caused pliron to spill 80/320 bytes to local memory for x4/x16
(`ptxas -v` confirmed "80/320 bytes cumulative stack size"). pliron generates
worse code for raw pointer dereferences than for bounds-checked slice
indexing — the opposite of normal Rust behavior. Reverted.

**Conclusion:** The x16 tiling (each thread handles 16 consecutive x-columns)
is fundamentally incompatible with efficient warp-level coalescing. The stride-16
access pattern to weight arrays generates cache pressure that offsets any benefit
from fewer threads. x4 is also slower than x1 (by ~1.2-1.35×) for the same
reason, just at smaller scale. cubecl's `#[unroll]` + JIT MLIR pass can
restructure the access pattern at compile time; cuda-oxide's AOT LLVM cannot.

**No further optimization attempts on tiled kernels are planned.** x1 is the
correct kernel for cuda-oxide's resize op. The tiled variants remain in the
codebase for completeness and cross-backend comparison purposes only.

**Implication for backend selection:** if tiled kernels matter for the
target workload (they likely don't for typical kornia preprocessing — x1
already wins everywhere), this is a real codegen gap that cubecl handles
cleanly and cuda-oxide does not yet.

## Fused pipeline P1 (resize → RGB→gray → normalize → HWC f32)

**Headline cubecl finding to test:** *adding gray+norm to a resize kernel
costs ~nothing when fused into one kernel.*

| Size | P0 resize-only (μs) | P1 sequential (μs) | P1 fused (μs) | **fused vs P0** |
|---|---:|---:|---:|---:|
| 1024² | 38.0 | 62.8 | 36.8 | **0.97×** (free, matches cubecl) |
| 2048² | 101.9 | 141.9 | 79.0 | **0.78×** — *fused faster than baseline* ⭐ |
| 4096² | 370.9 | 521.9 | 291.4 | **0.79×** — fused faster than baseline |
| 8K | 1511.5 | 2089.3 | 1134.6 | **0.75×** — fused faster than baseline |
| 1080p | 111.8 | 154.4 | 90.2 | **0.81×** — fused faster than baseline |

The fused kernel is **faster** than resize-alone at every size from 2048²
upwards. Likely root cause: P0's output is 3 unaligned u8 stores per dst
pixel; the fused gray output is 1 aligned 4-byte f32 store. The cleaner
store pattern dominates on GDDR6 even though the fused kernel does more
arithmetic. On cubecl/Tegra (unified memory) this difference didn't show.

**Sequential → fused speedup head-to-head:**

| Size | cubecl seq→fused | **cuda-oxide seq→fused** |
|---|---:|---:|
| 1024² | 1.80× | 1.71× |
| 2048² | 1.81× | **1.80×** |
| 4096² | 1.81× | **1.79×** |
| 8K | 1.81× | **1.84×** |
| 1080p | 1.66× | **1.71×** |

Both backends produce the **same qualitative finding** (sequential pays
~1.7-1.8× DRAM-roundtrip tax) and the magnitudes are tied.

## Fused pipeline P2 (resize → per-channel normalize → CHW f32)

The canonical ML preprocessing chain. Output is model-ready
(ResNet/ViT/YOLO).

| Size | P2 sequential (μs) | P2 fused (μs) | **seq → fused** |
|---|---:|---:|---:|
| 1024² | 47.9 | 36.3 | 1.32× |
| 2048² | 132.3 | 91.9 | 1.44× |
| 4096² | 509.0 | 340.9 | 1.49× |
| 8K | 2040.2 | 1380.4 | 1.48× |
| 1080p | 139.3 | 96.7 | 1.44× |

Compared to cubecl on this GPU (1.32-1.49× speedup): **tied within margin**.
Both backends find the same ~1.4-1.5× P2 fusion benefit.

## P2 fused vs P0 (full ML preprocessing vs just-the-resize)

| Size | P0 resize-only | P2 fused (CHW model-ready) | overhead vs P0 |
|---|---:|---:|---:|
| 1024² | 38.0 | 36.3 | 0.95× — fused *faster* |
| 2048² | 101.9 | 91.9 | 0.90× — fused *faster* |
| 4096² | 370.9 | 340.9 | 0.92× — fused *faster* |
| 8K | 1511.5 | 1380.4 | 0.91× — fused *faster* |
| 1080p | 111.8 | 96.7 | 0.86× — fused *faster* |

The **complete ML preprocessing pipeline runs faster than the resize step
alone** on cuda-oxide at every tested size. Same root cause as P1: cleaner
write pattern (3 separate f32 planes vs 1 packed u8x3 HWC interleaved) more
than offsets the extra arithmetic.

## Additional architectural gotcha — kernels calling slice-indexing helpers

The cubecl prototype's lib exposes three composable `#[cube]` primitives
(`sample_bilinear_u8_rgb_pixel`, `rgb_to_gray_u8`, `normalize_u8_to_f32`)
that fused kernels call inline. Tier-1 → Tier-3 composition pattern.

**On cuda-oxide today, this pattern triggers an Internal Compiler Error in
the codegen backend.** When a `#[kernel]` calls a non-intrinsic helper
function that indexes a slice parameter — regardless of whether the helper
is annotated `#[device]`, `#[inline(always)]`, or both — the pliron
MIR→LLVM conversion pass panics with `"Operation with use(s) being erased"`
at `mir-lower/src/convert/ops/call.rs:413`.

Workaround: inline the math manually into each fused kernel body. Each
fused kernel becomes one monolithic body. This works perf-wise (same
codegen result as if the helper had been inlined), but the **composable-
primitives ergonomic story is broken**.

Hashmap_v2's helpers (which work) take primitive args and return primitives.
Our resize helpers index slice params and return tuples — possibly a
trigger pattern combining slice-indexing inside the called function with
multi-value returns.

**Status (2026-05-28) — fixed upstream.** Tracked as NVlabs/cuda-oxide
issue #79 ("BUG: Tuple-returning device functions ICE during dialect-mir →
dialect-llvm lowering"), filed independently. Root cause confirmed: the
`is_unit` check in `mir-lower/src/convert/ops/call.rs` used
`is::<MirTupleType>()`, which matched every tuple including non-empty ones
like our helper's `(u32, u32, u32)` return — the result type was forced to
`void`, then `erase_operation` panicked because the MIR call still had
live uses (the destructured `(r, g, b)`). Commit `6ed9938`:

1. Tightened `is_unit` to fire only when the tuple has zero fields.
2. Hardened the `erase_operation` branch to return a `pliron::input_err`
   instead of letting the invariant panic — future misclassifications now
   surface as cuda-oxide diagnostics, not rustc "compiler unexpectedly
   panicked" banners. (Companion commit `396c76a` wraps device codegen in
   `catch_unwind` for the same reason.)

Verified locally against `origin/main` HEAD `396c76a` on 2026-05-28:
- `examples/ice_reproducer.rs` `with_helper_kernel` (multi-slice helper
  with `(u32, u32, u32)` tuple return) now compiles cleanly.
- Generated PTX (`ice_reproducer.ptx`) contains all three kernel entries
  including `with_helper_kernel` with an 86-line PTX body — not a stub.
- Toolchain unchanged: rustc nightly-2026-04-03, LLVM 21, sm_75.

**Implication for backend selection (unchanged):** the ICE itself is no
longer a blocker, so the composable Tier-1/Tier-2/Tier-3 pattern works on
cuda-oxide today. But the broader pattern — alpha-stage codegen pipeline
where MIR-call lowering bugs are a class of issues actively being shaken
out (75+ commits in two weeks, the catch_unwind wrapper acknowledges more
of these to come) — still argues for starting with cubecl per the mentor's
"face less blockers" steer. The proposal's Goal 2 / Goal 3 design should
still abstract behind a `Backend` trait so cuda-oxide can be slotted in
later once upstream stabilizes (or once Apple/AMD become non-goals).

## Still to do

- [ ] Correctness gate for fused kernels (currently bench-only; no
      bit-exact verification of P1/P2 outputs vs a CPU reference).
      Lower priority — the underlying ops are bit-exact verified
      individually.
- [ ] (Optional) AGX Orin or other Tegra reproduction — for direct cross-ref
      against the cubecl numbers in PR #897. Not blocking — discrete-GPU
      data already enough for Goal 5.
- [ ] (Optional) `_with_weights` pre-uploaded variant for cuda-oxide —
      we already pre-upload weights in the bench arms (weights live
      outside the timing loop), but no explicit API split.

## Current preliminary read

**On performance, baseline resize:** cuda-oxide is meaningfully faster than
cubecl at every size on discrete CUDA. 1.0-2.5× kernel-only,
2-7× end-to-end.

**On performance, fused pipelines:** **the headline cubecl finding holds on
cuda-oxide.** Fused P1 ≈ or BEATS resize-only cost at every size
(0.75-0.97× of P0). The CHW ML preprocessing fused pipeline also runs
faster than resize-alone (0.86-0.95× of P0). Sequential→fused speedups are
within margin of cubecl on the same GPU (1.7-1.84× P1, 1.3-1.5× P2).

**On performance, tiled kernels:** cubecl wins decisively. After exhaustive
optimization (seq-macro unrolling, LLVM flags, get_unchecked), cuda-oxide's
x16 remains 20-30× slower than its x1 (vs cubecl's 1.5-3× slower). Root
cause is architectural: the x16 tiling pattern generates stride-16 warp
accesses to weight arrays (32× more cache lines than x1's stride-1), combined
with 226 bounds-check branches that serialize memory loads. Not a
loop-unrolling issue — the loops are fully eliminated from PTX, but the
fundamental coalescing mismatch remains. Does not matter for production (x1
wins everywhere on CUDA anyway) but is a real codegen quality gap vs cubecl.

**On cross-platform:** cubecl is unambiguously broader (CUDA + ROCm + WGPU +
CPU-MLIR + Metal vs cuda-oxide's CUDA-only). For a library that aspires to
run anywhere kornia-rs runs today, this is a heavy thumb on cubecl's side
of the scale.

**On composability ergonomics:** tied (was a cuda-oxide loss, now resolved).
The helper-call ICE that previously forced monolithic kernel bodies on
cuda-oxide was fixed upstream by commit `6ed9938` (issue #79) and verified
locally on 2026-05-28 — the cubecl-style "Tier-1 primitives → Tier-2
launchers → Tier-3 fused pipelines" pattern now works on both backends.
The lib.rs in this prototype still ships monolithic kernels (the workaround)
because the bake-off was already complete by the time the fix landed;
refactoring to use helpers is a low-risk future cleanup.

**On day-to-day ergonomics:** cuda-oxide wins. Single-source compilation,
standard Rust syntax (no `#[cube]` proc-macro DSL), rust-analyzer works
normally, `cuda-gdb` integration. cubecl's DSL is the cost paid for
cross-platform-ness.

**On integration complexity:** comparable in friction, different in nature
(cubecl's friction is in the Rust dep graph; cuda-oxide's is in the system
toolchain — LLVM 21 + clang 21 + nightly + custom rustc backend).

### Bottom line for Goal 5 (backend selection)

The matrix of trade-offs is more textured than the resize-only data
suggested. Honest summary:

**If the project deems CUDA-only acceptable** (with a separate CPU
fallback strategy for non-CUDA targets):
- cuda-oxide gives meaningfully better perf on the baseline op (1-2.5×)
- The headline fusion finding holds and even improves
- Helper-call ICE fixed upstream (issue #79, 2026-05-27); composable
  primitives now work on both backends
- But tiled kernels are still dramatically worse than cubecl's, and the
  toolchain remains nightly + LLVM 21 + custom rustc backend
- Active codegen churn (75+ commits in two weeks) — `catch_unwind` wrapper
  in commit `396c76a` acknowledges more MIR-lowering bugs likely

**If multi-backend portability matters** (kornia-rs on AMD GPU, M-series
Mac, browser-deployed wgpu, or aarch64 CPU fallback that doesn't go
through NEON):
- cubecl is the only choice — cuda-oxide is CUDA-only by design
- Performance on Jetson Orin (PR #897) is genuinely good
- The `#[cube]` DSL ergonomic cost is the price paid

**My read:** the cross-platform requirement is what should drive the
decision. If kornia-rs already commits to a wgpu / Metal / ROCm strategy
in the proposal, cubecl is the answer despite cuda-oxide's CUDA-only perf
advantages. If the GpuAllocator API surface can stay CUDA-only (with the
existing CPU code path handling everything else), cuda-oxide becomes
viable but the team should be prepared to file/fix codegen bugs upstream.

Either way, **Goals 1-4 (GpuAllocator, tensor-op traits, kornia-imgproc
dispatch) should not start until this decision lands** — the trait
machinery of GpuAllocator<B: Backend> depends on which backend's runtime
types it binds.
