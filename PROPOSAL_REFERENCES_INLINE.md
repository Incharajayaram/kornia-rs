# Proposal References (Inline-Citation Version)

This document rewrites the key technical claims in proposal style with inline references, using the Section 6 source list from the draft.

## 1. Problem Framing

`kornia-rs` is positioned as a low-level, performance-oriented Rust computer vision stack, and the current architecture already exposes allocator-parametric tensors (`Tensor<T, N, A>`), which makes backend extension feasible without redesigning public APIs [1](https://arxiv.org/abs/2505.12425) [3](https://github.com/kornia/kornia-rs) [4](https://github.com/kornia/kornia-rs/blob/main/crates/kornia-tensor/src/allocator.rs). The motivation for GPU acceleration is aligned with common pixel-parallel workloads (`resize`, warp operations, color transforms), where throughput becomes a bottleneck under high-resolution and real-time pipelines [1](https://arxiv.org/abs/2505.12425) [6](https://github.com/kornia/kornia-rs/blob/main/crates/kornia-imgproc/src/resize.rs).

## 2. Allocator and Storage Safety Basis

The allocator abstraction is already present in `kornia-tensor`, but storage paths currently assume host-accessible memory in several places; this is why introducing a GPU allocator requires domain-safe storage and explicit transfer semantics before kernel dispatch [4](https://github.com/kornia/kornia-rs/blob/main/crates/kornia-tensor/src/allocator.rs) [5](https://github.com/kornia/kornia-rs/blob/main/crates/kornia-tensor/src/storage.rs). The design follows Rust allocation/layout constraints and strict ownership/deallocation pairing [15](https://doc.rust-lang.org/std/alloc/trait.Allocator.html) [16](https://doc.rust-lang.org/std/alloc/struct.Layout.html).

## 3. Backend Choice and Integration Direction

CubeCL is selected as the primary backend because it supports Rust-native kernel authoring and multi-runtime targeting, which better matches the allocator-dispatch architecture being introduced [8](https://github.com/tracel-ai/cubecl) [9](https://docs.rs/cubecl/latest/cubecl/) [10](https://github.com/tracel-ai/cubecl/tree/main/examples). The integration pattern is also consistent with prior ecosystem practice in Burn’s CubeCL backend [11](https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl).

Native CUDA interop remains the defined fallback path when backend criteria are not met, using `cust`/Rust-CUDA ecosystem tooling [13](https://docs.rs/cust/latest/cust/) [14](https://github.com/Rust-GPU/Rust-CUDA) [18](https://docs.nvidia.com/cuda/).

## 4. Explicit Memory Movement and Heterogeneous Execution

The proposal enforces explicit host-device movement (`to_device`, `to_host`) and avoids implicit transfer behavior inside high-level operations, which is consistent with explicit-memory GPU execution models and avoids hidden synchronization/latency costs [8](https://github.com/tracel-ai/cubecl) [18](https://docs.nvidia.com/cuda/) [12](https://www.vectorware.com/blog/rust-std-on-gpu/).

## 5. Validation Methodology

Correctness and compatibility are validated with CPU-vs-GPU parity and compile-fail boundary checks (for host-only API gating on device tensors), with `trybuild` as the compile-fail test infrastructure [17](https://docs.rs/trybuild/latest/trybuild/) [4](https://github.com/kornia/kornia-rs/blob/main/crates/kornia-tensor/src/allocator.rs) [5](https://github.com/kornia/kornia-rs/blob/main/crates/kornia-tensor/src/storage.rs). Performance validation is benchmark-layered and grounded in kernel/runtime metrics from CubeCL and CUDA tooling [9](https://docs.rs/cubecl/latest/cubecl/) [13](https://docs.rs/cust/latest/cust/) [18](https://docs.nvidia.com/cuda/).

## 6. Scope Alignment With Program Call

The implementation scope directly matches the GSoC idea call for GPU acceleration in `kornia-tensor` and `kornia-imgproc`, with backend choice and staged implementation tied to the project idea constraints [7](https://github.com/kornia/kornia-rs/wiki/%5B2026%5D-Google-Sumer-of-Code-Application).

---

## Full Reference List (Section 6)

1. E. Riba, J. Shi, A. Kumar, A. Shen, G. Bradski. Kornia-rs: A Low-Level 3D Computer Vision Library In Rust: arXiv:2505.12425, 2025. https://arxiv.org/abs/2505.12425
2. E. Riba, D. Mishkin, D. Ponsa, E. Rublee, G. Bradski. Kornia: An Open Source Differentiable Computer Vision Library for PyTorch. WACV 2020: https://arxiv.org/abs/1910.02190
3. kornia-rs repository: https://github.com/kornia/kornia-rs
4. kornia-rs TensorAllocator source: https://github.com/kornia/kornia-rs/blob/main/crates/kornia-tensor/src/allocator.rs
5. kornia-rs TensorStorage source: https://github.com/kornia/kornia-rs/blob/main/crates/kornia-tensor/src/storage.rs
6. kornia-rs imgproc resize implementation: https://github.com/kornia/kornia-rs/blob/main/crates/kornia-imgproc/src/resize.rs
7. GSoC 2026 project idea: GPU acceleration for kornia-tensor and kornia-imgproc: https://github.com/kornia/kornia-rs/wiki/%5B2026%5D-Google-Sumer-of-Code-Application
8. CubeCL repository: https://github.com/tracel-ai/cubecl
9. CubeCL crate documentation: https://docs.rs/cubecl/latest/cubecl/
10. CubeCL examples (matmul, reduce): https://github.com/tracel-ai/cubecl/tree/main/examples
11. Burn CubeCL backend, direct architectural analog to GpuAllocator dispatch model. https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl
12. Vectorware blog: Rust std on GPU https://www.vectorware.com/blog/rust-std-on-gpu/
13. cust crate documentation: https://docs.rs/cust/latest/cust/
14. Rust-CUDA project https://github.com/Rust-GPU/Rust-CUDA
15. Rust Allocator trait (std::alloc): https://doc.rust-lang.org/std/alloc/trait.Allocator.html
16. Rust Layout type (std::alloc::Layout): https://doc.rust-lang.org/std/alloc/struct.Layout.html
17. trybuild crate; compile-fail test infrastructure used in validation layer: https://docs.rs/trybuild/latest/trybuild/
18. NVIDIA CUDA Toolkit documentation. https://docs.nvidia.com/cuda/
