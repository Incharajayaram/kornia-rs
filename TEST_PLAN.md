# TEST_PLAN.md

## Test Audit & Expansion Plan for kornia-tensor and kornia-imgproc

**Goal:**
Establish comprehensive, robust unit test coverage for `kornia-tensor` and `kornia-imgproc` as groundwork for future GPU backend support. This plan documents current coverage, identifies gaps, and proposes new tests for edge cases, error handling, and API invariants.

---

### 1. Current Test Coverage (Audit)

#### kornia-tensor
- [x] Tensor creation (various shapes, dtypes)
- [x] Reshape, permute, contiguous
- [x] Zeros, ones, map, get/set
- [x] Storage: from_vec, into_vec, buffer mutability
- [x] Indexing (checked/unchecked)
- [x] Stride/layout correctness
- [x] Error handling (some)
- [ ] Empty tensor edge cases
- [ ] Max/min size tensors
- [ ] Invalid shape/allocator errors
- [ ] Round-trip conversions (tensor <-> image)

#### kornia-imgproc
- [x] Resize (smoke, ch1/ch3, meshgrid)
- [x] Core ops: std_mean, bitwise, hconcat
- [x] Warp perspective
- [x] Filter kernels (sobel, gaussian, box blur)
- [x] Metrics (l1, huber)
- [x] Contours, features (partial)
- [ ] Error handling (invalid input, unsupported interpolation)
- [ ] Boundary conditions (image borders, 1x1, 0x0)
- [ ] API invariants (output shape, dtype)

---

### 2. Gaps & Areas for Improvement
- Missing tests for empty tensors/images
- No explicit tests for max/min size or invalid shapes
- Error handling for allocators and invalid input is incomplete
- Some API invariants (e.g., round-trip conversions, stride correctness) not directly tested
- Boundary conditions (e.g., 1x1, 0x0 images) under-tested

---

### 3. Proposed New Test Cases

#### kornia-tensor
- `test_empty_tensor_creation()`
- `test_tensor_max_size()`
- `test_tensor_invalid_shape()`
- `test_allocator_error_handling()`
- `test_tensor_roundtrip_image_conversion()`
- `test_stride_and_layout_invariants()`

#### kornia-imgproc
- `test_resize_empty_image()`
- `test_resize_max_size_image()`
- `test_resize_invalid_input()`
- `test_filter_boundary_conditions()`
- `test_metrics_invalid_input()`
- `test_core_ops_output_shape_dtype()`

---

### 4. Rationale
- These tests will catch regressions and undefined behavior as new allocators and device backends are introduced.
- They provide a safety net for future refactors and feature additions.
- They help maintainers and contributors validate correctness and stability.

---

### 5. Next Steps
- Implement new test cases in local branch (do not PR until assigned)
- Run full test suite locally (`pixi run rust-test`)
- Update this plan as new gaps are discovered or as maintainers provide feedback

---

**Prepared by:**
[Your Name]
14 March 2026
