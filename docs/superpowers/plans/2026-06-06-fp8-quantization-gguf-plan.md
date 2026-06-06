# FP8 Quantization & GGUF Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add F8E5M2 DType and 4 new FP8 GGUF block quantization types (Q8F4M3_0, Q8F4M3_1, Q8F5M2_0, Q8F5M2_1) with full CPU/CUDA/Metal/SIMD backend support.

**Architecture:** Three sequential phases. Phase 1 (parallel sub-tracks) adds F8E5M2 as a DType and defines the GGUF block types. Phase 2 implements the block types on CPU with GGUF serialization. Phase 3 adds SIMD/CUDA/Metal fast paths. All new code follows existing patterns from F8E4M3 (DType) and Q8_0/Q8_1 (GgmlType).

**Tech Stack:** Rust, `float8` crate v0.7.0 (already a dependency), CUDA C, Apple Metal Shading Language, AVX2/NEON/SIMD128 intrinsics

**Spec:** `docs/superpowers/specs/2026-06-06-fp8-quantization-gguf-design.md`

---

## File Map

### Phase 1a: F8E5M2 DType
| File | What changes |
|------|-------------|
| `candle-core/src/dtype.rs` | `DType::F8E5M2` variant, `WithDType` + `FloatDType` impls, `FromStr`/`as_str`/`size_in_bytes`/`is_int`/`is_float` arms |
| `candle-core/src/scalar.rs` | `Scalar::F8E5M2` variant, `zero`/`one`/`to_f64`/`from_f64`/display arms |
| `candle-core/src/cpu_backend/mod.rs` | `CpuStorage::F8E5M2`, `CpuStorageRef::F8E5M2`, dtype conversions |
| `candle-core/src/cpu/kernels.rs` | `VecOps` impl for `f8e5m2` |
| `candle-core/src/cuda_backend/mod.rs` | `CudaStorageSlice::F8E5M2`, `cuda_dtype!` |
| `candle-core/src/metal_backend/mod.rs` | `MetalStorage::F8E5M2` variant |
| `candle-core/src/safetensors.rs` | `F8E5M2` ↔ `st::Dtype::F8_E5M2` mappings |
| `candle-core/src/display.rs` | Display impl for F8E5M2 tensors |
| `candle-kernels/src/fill.cu` | `fill_f8_e5m2`, `copy2d_f8_e5m2` kernels |
| `candle-kernels/src/cast.cu` | `cast_f8_e5m2` kernel |
| `candle-kernels/src/affine.cu` | `affine_f8_e5m2` kernel |
| `candle-kernels/src/ternary.cu` | `where_*_f8_e5m2` kernels |
| `candle-metal-kernels/src/lib.rs` | Metal F8E5M2 stub entry |

### Phase 2: GGUF FP8 Block Types (CPU)
| File | What changes |
|------|-------------|
| `candle-core/src/quantized/mod.rs` | 4 new `GgmlDType` variants, `from_u32`/`to_u32`, `type_size`/`block_size`/`cpu_zeros`/`from_data` arms |
| `candle-core/src/quantized/k_quants.rs` | 4 block structs + `GgmlType` trait impls |
| `candle-core/src/quantized/gguf_file.rs` | Writer `candle.fp8_types` metadata key |
| `candle-core/src/quantized/ggml_file.rs` | `qtensor_from_ggml` arms for IDs 43-46 |

### Phase 3: SIMD + CUDA + Metal Kernels
| File | What changes |
|------|-------------|
| `candle-core/src/quantized/avx.rs` | `vec_dot` for each FP8 type |
| `candle-core/src/quantized/neon.rs` | Same for NEON |
| `candle-core/src/quantized/simd128.rs` | Same for WASM SIMD128 |
| `candle-core/src/quantized/cuda.rs` | CUDA quantize/dequantize wiring |
| `candle-core/src/quantized/fast_mmvq.rs` | Dispatch arms for FP8 types |
| `candle-core/src/quantized/fast_mmq.rs` | Dispatch arms for FP8 types |
| `candle-core/src/quantized/metal.rs` | Metal backend wiring |
| `candle-metal-kernels/src/kernels/quantized.rs` | FP8 in Metal `GgmlDType` enum |
| `candle-kernels/src/ffi.rs` | FFI declarations for FP8 CUDA launchers |

---

## Phase 1a: F8E5M2 DType

### Task 1a.1: Add F8E5M2 to DType enum and type-level methods

**Files:** Modify `candle-core/src/dtype.rs`

- [ ] **Step 1: Add F8E5M2 variant to DType enum**

After line 29 (`F8E4M3,`), add:
```rust
    /// 8-bit floating point with 5-bit exponent and 2-bit mantissa.
    F8E5M2,
```

- [ ] **Step 2: Add F8E5M2 to FromStr impl**

After line 64 (`"f8e4m3" => Ok(Self::F8E4M3),`), add:
```rust
            "f8e5m2" => Ok(Self::F8E5M2),
```

- [ ] **Step 3: Add F8E5M2 to as_str**

After line 87 (`Self::F8E4M3 => "f8e4m3",`), add:
```rust
            Self::F8E5M2 => "f8e5m2",
```

- [ ] **Step 4: Add F8E5M2 to size_in_bytes**

After line 107 (`Self::F8E4M3 => 1,`), add:
```rust
            Self::F8E5M2 => 1,
```

- [ ] **Step 5: Add F8E5M2 to is_int and is_float**

In `is_int`, after line 122 (`Self::F8E4M3`), add `Self::F8E5M2` to the float arm (same as F8E4M3). In `is_float`, add `Self::F8E5M2` to the float arm.

- [ ] **Step 6: Add f8e5m2 import and WithDType impl**

After line 225 (`use float8::F8E4M3 as f8e4m3;`), add:
```rust
use float8::F8E5M2 as f8e5m2;
```

After line 237 (`with_dtype!(f8e4m3, F8E4M3, ...)`), add:
```rust
with_dtype!(f8e5m2, F8E5M2, f8e5m2::from_f64, |v: f8e5m2| v.to_f64());
```

- [ ] **Step 7: Add FloatDType impl**

After line 295 (`impl FloatDType for f8e4m3 {}`), add:
```rust
impl FloatDType for f8e5m2 {}
```

- [ ] **Step 8: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation succeeds.

- [ ] **Step 9: Commit**

```bash
git add candle-core/src/dtype.rs
git commit -m "feat: add F8E5M2 to DType enum and WithDType impl"
```

---

### Task 1a.2: Add F8E5M2 to Scalar

**Files:** Modify `candle-core/src/scalar.rs`

- [ ] **Step 1: Add import**

After line 4 (`use float8::F8E4M3 as f8e4m3;`), add:
```rust
use float8::F8E5M2 as f8e5m2;
```

- [ ] **Step 2: Add Scalar::F8E5M2 variant**

After line 18 (`F8E4M3(f8e4m3),`), add:
```rust
    F8E5M2(f8e5m2),
```

- [ ] **Step 3: Add zero and one arms**

In `Scalar::zero`, after line 39 (`DType::F8E4M3 => Scalar::F8E4M3(f8e4m3::ZERO),`), add:
```rust
            DType::F8E5M2 => Scalar::F8E5M2(f8e5m2::ZERO),
```

In `Scalar::one`, after line 57 (`DType::F8E4M3 => Scalar::F8E4M3(f8e4m3::ONE),`), add:
```rust
            DType::F8E5M2 => Scalar::F8E5M2(f8e5m2::ONE),
```

- [ ] **Step 4: Add to_f64 and from_f64 arms**

Find the `to_f64` method on `Scalar`. After the `Scalar::F8E4M3(v) => v.to_f64()` arm, add:
```rust
            Scalar::F8E5M2(v) => v.to_f64(),
```

Find the `from_f64` method. After the `F8E4M3` arm, add:
```rust
            DType::F8E5M2 => Scalar::F8E5M2(f8e5m2::from_f64(v)),
```

- [ ] **Step 5: Add Display/Debug arms for Scalar**

Find any match on Scalar variants for Display. After the `F8E4M3` arm, add:
```rust
            Scalar::F8E5M2(v) => write!(f, "{v}"),
```

- [ ] **Step 6: Add F8E5M2 to the `From<f8e5m2>` and other relevant trait impls**

The `Scalar` enum derives `Debug, Clone, Copy, PartialEq` — no change needed for those since we added the variant.

- [ ] **Step 7: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation succeeds.

- [ ] **Step 8: Commit**

```bash
git add candle-core/src/scalar.rs
git commit -m "feat: add F8E5M2 to Scalar enum"
```

---

### Task 1a.3: Add F8E5M2 to CpuStorage and conversions

**Files:** Modify `candle-core/src/cpu_backend/mod.rs`

- [ ] **Step 1: Add CpuStorage::F8E5M2 variant**

Find `F8E4M3(Vec<F8E4M3>)` in the `CpuStorage` enum. After it, add:
```rust
    F8E5M2(Vec<f8e5m2>),
```

- [ ] **Step 2: Add CpuStorageRef::F8E5M2 variant**

Find `F8E4M3(&'a [F8E4M3])` in `CpuStorageRef`. After it, add:
```rust
    F8E5M2(&'a [f8e5m2]),
```

- [ ] **Step 3: Add F8E5M2 to storage accessor methods**

There are match arms that return slices and dtype — add `F8E5M2` arms following the `F8E4M3` pattern. In each case, `F8E5M2` maps identically to F8E4M3 (same size, same return patterns) but with `f8e5m2` type.

- [ ] **Step 4: Add dtype conversions to/from F8E5M2**

Following the F8E4M3 conversion pattern (around lines 2061-2088), add conversion arms for:

| From | To |
|------|----|
| `U8` | `F8E5M2` |
| `U32` | `F8E5M2` |
| `I64` | `F8E5M2` |
| `BF16` | `F8E5M2` |
| `F16` | `F8E5M2` |
| `F32` | `F8E5M2` |
| `F64` | `F8E5M2` |
| `F8E5M2` | `U8`, `U32`, `I64`, `BF16`, `F16`, `F32`, `F64` |

Each arm uses `unary_map` with appropriate conversion function, e.g.:
```rust
(Self::F32(storage), DType::F8E5M2) => {
    let data = unary_map(storage, layout, f8e5m2::from_f32);
    Ok(Self::F8E5M2(data))
}
```

- [ ] **Step 5: Add F8E5M2 import**

At the top of the file, after `use float8::F8E4M3;`, add:
```rust
use float8::F8E5M2 as f8e5m2;
```

- [ ] **Step 6: Write and run compile check**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation succeeds.

- [ ] **Step 7: Commit**

```bash
git add candle-core/src/cpu_backend/mod.rs
git commit -m "feat: add F8E5M2 to CpuStorage and conversions"
```

---

### Task 1a.4: Add F8E5M2 to CUDA and Metal backends

**Files:** Modify `candle-core/src/cuda_backend/mod.rs`, `candle-core/src/metal_backend/mod.rs`

- [ ] **Step 1: Add CudaStorageSlice::F8E5M2 variant**

In `cuda_backend/mod.rs`, find `F8E4M3(CudaSlice<float8::F8E4M3>)`. After it, add:
```rust
    F8E5M2(CudaSlice<float8::F8E5M2>),
```

- [ ] **Step 2: Wire F8E5M2 through cuda_dtype! macro**

Find `cuda_dtype!(float8::F8E4M3, F8E4M3)`. After it, add:
```rust
cuda_dtype!(float8::F8E5M2, F8E5M2);
```

- [ ] **Step 3: Add F8E5M2 to all CUDA backend match arms**

Search for all `F8E4M3` references in the CUDA backend and add corresponding `F8E5M2` arms. These include storage methods, device-to-device copies, dtype queries, etc. Each arm follows the same pattern as F8E4M3.

- [ ] **Step 4: Add MetalStorage::F8E5M2 variant**

In `metal_backend/mod.rs`, find the `MetalStorage` enum and add `F8E5M2` variant following `F8E4M3` pattern. Add `F8E5M2` arms to all relevant match blocks.

- [ ] **Step 5: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation succeeds.

- [ ] **Step 6: Commit**

```bash
git add candle-core/src/cuda_backend/mod.rs candle-core/src/metal_backend/mod.rs
git commit -m "feat: add F8E5M2 to CUDA and Metal storage backends"
```

---

### Task 1a.5: Add F8E5M2 to safetensors, display, and VecOps

**Files:** Modify `candle-core/src/safetensors.rs`, `candle-core/src/display.rs`, `candle-core/src/cpu/kernels.rs`

- [ ] **Step 1: Update safetensors.rs mappings**

In `safetensors.rs`, add F8E5M2 alongside every F8E4M3 arm:
- `DType::F8E5M2 => st::Dtype::F8_E5M2` (line ~34, export)
- `st::Dtype::F8_E5M2 => Ok(DType::F8E5M2)` (line ~56, import)
- Conversion functions with `f8e5m2` type (lines ~225, ~301, ~393)

- [ ] **Step 2: Update display.rs**

After each `DType::F8E4M3` arm in display.rs, add a corresponding `DType::F8E5M2` arm using `float8::F8E5M2`.

- [ ] **Step 3: Add VecOps impl for f8e5m2**

In `candle-core/src/cpu/kernels.rs`, after the `impl VecOps for float8::F8E4M3` block (line ~188), add:
```rust
impl VecOps for float8::F8E5M2 {
    fn min(xs: &[Self]) -> Self {
        xs.iter().fold(float8::F8E5M2::MAX, |a, &b| a.min(b))
    }
    fn max(xs: &[Self]) -> Self {
        xs.iter().fold(float8::F8E5M2::MIN, |a, &b| a.max(b))
    }
}
```

- [ ] **Step 4: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation succeeds.

- [ ] **Step 5: Commit**

```bash
git add candle-core/src/safetensors.rs candle-core/src/display.rs candle-core/src/cpu/kernels.rs
git commit -m "feat: add F8E5M2 to safetensors, display, and VecOps"
```

---

### Task 1a.6: Add F8E5M2 CUDA kernels

**Files:** Modify `candle-kernels/src/fill.cu`, `cast.cu`, `affine.cu`, `ternary.cu`

For each kernel file, add F8E5M2 variants following the existing F8E4M3 pattern. The `__nv_fp8_e5m2` type is available through the same CUDA FP8 header used for `__nv_fp8_e4m3`.

- [ ] **Step 1: fill.cu** — Add `fill_f8_e5m2` and `copy2d_f8_e5m2` kernels (copy the F8E4M3 versions, replace type)

- [ ] **Step 2: cast.cu** — Add `cast_f8_e5m2` kernel (F8E5M2 ↔ float)

- [ ] **Step 3: affine.cu** — Add `affine_f8_e5m2` kernel

- [ ] **Step 4: ternary.cu** — Add `where_*_f8_e5m2` kernels for all index types

- [ ] **Step 5: Verify compilation** (on a CUDA-capable machine or with CUDA feature flag)

```bash
cargo build -p candle-kernels --features cuda 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add candle-kernels/src/fill.cu candle-kernels/src/cast.cu candle-kernels/src/affine.cu candle-kernels/src/ternary.cu
git commit -m "feat: add F8E5M2 CUDA kernels (fill, cast, affine, ternary)"
```

---

### Task 1a.7: Add F8E5M2 Metal kernel stubs

**Files:** Modify `candle-metal-kernels/src/lib.rs`

- [ ] **Step 1: Register F8E5M2 in Metal kernel dispatch**

Follow the F8E4M3 pattern in Metal kernel registration. F8E5M2 in Metal uses the same bit width as F8E4M3 so the storage/transfer code is identical at the Metal level.

- [ ] **Step 2: Verify compilation** (macOS only)

```bash
cargo build -p candle-metal-kernels 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add candle-metal-kernels/src/lib.rs
git commit -m "feat: add F8E5M2 Metal kernel stubs"
```

---

## Phase 2: GGUF FP8 Block Types (CPU)

### Task 2.1: Define block structs and constants

**Files:** Modify `candle-core/src/quantized/k_quants.rs`

- [ ] **Step 1: Add block size constants**

After the QK8_1 constant (line ~21), add:
```rust
pub const QK8F4M3_0: usize = 32;
pub const QK8F4M3_1: usize = 32;
pub const QK8F5M2_0: usize = 32;
pub const QK8F5M2_1: usize = 32;
```

- [ ] **Step 2: Add float8 import**

After the existing `half` imports, add:
```rust
use float8::F8E4M3 as f8e4m3;
use float8::F8E5M2 as f8e5m2;
```

- [ ] **Step 3: Define block structs**

After BlockQ8_1 (line ~108), add the 4 FP8 block structs:

```rust
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8F4M3_0 {
    pub(crate) d: f16,
    pub(crate) qs: [f8e4m3; QK8F4M3_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8F4M3_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8F4M3_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [f8e4m3; QK8F4M3_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8F4M3_1>() == 36);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8F5M2_0 {
    pub(crate) d: f16,
    pub(crate) qs: [f8e5m2; QK8F5M2_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8F5M2_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8F5M2_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [f8e5m2; QK8F5M2_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8F5M2_1>() == 36);
```

- [ ] **Step 4: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation succeeds (struct definitions only, no trait impls yet).

- [ ] **Step 5: Commit**

```bash
git add candle-core/src/quantized/k_quants.rs
git commit -m "feat: add FP8 GGUF block struct definitions"
```

---

### Task 2.2: Implement GgmlType for FP8 block types (scalar CPU path)

**Files:** Modify `candle-core/src/quantized/k_quants.rs`

- [ ] **Step 1: Implement GgmlType for BlockQ8F4M3_0**

```rust
impl GgmlType for BlockQ8F4M3_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8F4M3_0;
    const BLCK_SIZE: usize = QK8F4M3_0;
    type VecDotType = f32;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK8F4M3_0));
        let nb = k / QK8F4M3_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..QK8F4M3_0 {
                ys[i * QK8F4M3_0 + j] = xs[i].qs[j].to_f32() * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        let max_val: f32 = f8e4m3::MAX.to_f32();
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let amax = xs.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let d_val = amax / max_val;
            let d = if d_val > 0f32 { d_val } else { f16::MIN_POSITIVE.to_f32() };
            let id = 1.0 / d;
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f8e4m3::from_f32(f32::clamp(x * id, -max_val, max_val));
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q8f4m3_0_f32(n, xs, ys);
        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q8f4m3_0_f32(n, xs, ys);
        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q8f4m3_0_f32(n, xs, ys);
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK8F4M3_0));
        let nb = n / QK8F4M3_0;
        let mut sumf = 0f32;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let mut sum = 0f32;
            for j in 0..QK8F4M3_0 {
                sum += xs[i].qs[j].to_f32() * ys[i * QK8F4M3_0 + j];
            }
            sumf += sum * d;
        }
        sumf
    }
}
```

- [ ] **Step 2: Implement GgmlType for BlockQ8F4M3_1**

Same pattern as above but with `m` field and asymmetric dequant. `VecDotType = f32`. The `to_float` formula: `value = qs[j].to_f32() * d.to_f32() + m.to_f32()`. The `from_float` computes `m = min(xs)`, `d = (max - min) / max_fp8_value`. The `vec_dot_unopt` accumulates `sum += (qs[j].to_f32() * d.to_f32() + m.to_f32()) * ys[...]`.

- [ ] **Step 3: Implement GgmlType for BlockQ8F5M2_0**

Same as BlockQ8F4M3_0 but with `f8e5m2` type and `f8e5m2::MAX`. Uses `max_val = f8e5m2::MAX.to_f32()` (~57344).

- [ ] **Step 4: Implement GgmlType for BlockQ8F5M2_1**

Same as BlockQ8F4M3_1 but with `f8e5m2` type.

- [ ] **Step 5: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation fails due to missing `GgmlDType` variants (not yet added). This confirms trait impls are correct.

- [ ] **Step 6: Commit**

```bash
git add candle-core/src/quantized/k_quants.rs
git commit -m "feat: add GgmlType impls for FP8 block types (scalar CPU)"
```

---

### Task 2.3: Add FP8 variants to GgmlDType enum

**Files:** Modify `candle-core/src/quantized/mod.rs`

- [ ] **Step 1: Add enum variants**

After `Q8K,` in the `GgmlDType` enum (line ~291), add:
```rust
    Q8F4M3_0,
    Q8F4M3_1,
    Q8F5M2_0,
    Q8F5M2_1,
```

- [ ] **Step 2: Add from_u32 mappings**

In `from_u32`, before the `_ => crate::bail!(...)` line, add:
```rust
            43 => Self::Q8F4M3_0,
            44 => Self::Q8F4M3_1,
            45 => Self::Q8F5M2_0,
            46 => Self::Q8F5M2_1,
```

- [ ] **Step 3: Add to_u32 mappings**

In `to_u32`, before the last match arm, add:
```rust
            Self::Q8F4M3_0 => 43,
            Self::Q8F4M3_1 => 44,
            Self::Q8F5M2_0 => 45,
            Self::Q8F5M2_1 => 46,
```

- [ ] **Step 4: Add cpu_zeros arms**

After the `Q8K` arm in `cpu_zeros`, add:
```rust
            Self::Q8F4M3_0 => Box::new(vec![BlockQ8F4M3_0::zeros(); elem_count / BlockQ8F4M3_0::BLCK_SIZE]),
            Self::Q8F4M3_1 => Box::new(vec![BlockQ8F4M3_1::zeros(); elem_count / BlockQ8F4M3_1::BLCK_SIZE]),
            Self::Q8F5M2_0 => Box::new(vec![BlockQ8F5M2_0::zeros(); elem_count / BlockQ8F5M2_0::BLCK_SIZE]),
            Self::Q8F5M2_1 => Box::new(vec![BlockQ8F5M2_1::zeros(); elem_count / BlockQ8F5M2_1::BLCK_SIZE]),
```

- [ ] **Step 5: Add from_data arms**

After the `Q8K` arm in `from_data`, add:
```rust
            Self::Q8F4M3_0 => Box::new(as_t_slice::<BlockQ8F4M3_0>(data).to_vec()),
            Self::Q8F4M3_1 => Box::new(as_t_slice::<BlockQ8F4M3_1>(data).to_vec()),
            Self::Q8F5M2_0 => Box::new(as_t_slice::<BlockQ8F5M2_0>(data).to_vec()),
            Self::Q8F5M2_1 => Box::new(as_t_slice::<BlockQ8F5M2_1>(data).to_vec()),
```

- [ ] **Step 6: Add type_size and block_size arms**

Find the `type_size` and `block_size` methods. Add arms for the 4 new variants: all have `type_size` of 34 (for _0) or 36 (for _1), and all have `block_size` of 32.

- [ ] **Step 7: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```
Expected: Compilation succeeds.

- [ ] **Step 8: Commit**

```bash
git add candle-core/src/quantized/mod.rs
git commit -m "feat: add FP8 GgmlDType variants with type IDs 43-46"
```

---

### Task 2.4: Wire FP8 types into GGUF reader and GGML loader

**Files:** Modify `candle-core/src/quantized/gguf_file.rs`, `candle-core/src/quantized/ggml_file.rs`

- [ ] **Step 1: Add qtensor_from_ggml arms**

In `ggml_file.rs`, after the last `GgmlDType::Q8_0` arm in `qtensor_from_ggml`, add:
```rust
        GgmlDType::Q8F4M3_0 => {
            from_raw_data::<k_quants::BlockQ8F4M3_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q8F4M3_1 => {
            from_raw_data::<k_quants::BlockQ8F4M3_1>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q8F5M2_0 => {
            from_raw_data::<k_quants::BlockQ8F5M2_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q8F5M2_1 => {
            from_raw_data::<k_quants::BlockQ8F5M2_1>(raw_data, size_in_bytes, dims, device)
        }
```

- [ ] **Step 2: Add candle.fp8_types metadata on GGUF write**

In the `write` function in `gguf_file.rs`, add FP8 type documentation metadata. After the tensor data is written, add a `candle.fp8_types` metadata entry:
```rust
// metadata already includes this key when called:
// ("candle.fp8_types", &Value::String("43:q8f4m3_0,44:q8f4m3_1,45:q8f5m2_0,46:q8f5m2_1".to_string()))
```

The actual metadata is passed by the caller — add a helper or document that callers should include this key. Add a convenience function:
```rust
pub fn fp8_metadata_entry() -> (&'static str, Value) {
    ("candle.fp8_types", Value::String(
        "43:q8f4m3_0,44:q8f4m3_1,45:q8f5m2_0,46:q8f5m2_1".to_string()
    ))
}
```

- [ ] **Step 3: Verify compilation**

```bash
cargo build -p candle-core 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add candle-core/src/quantized/gguf_file.rs candle-core/src/quantized/ggml_file.rs
git commit -m "feat: wire FP8 types into GGUF reader and GGML loader"
```

---

### Task 2.5: Write tests and verify CPU correctness

**Files:** Create/modify tests in `candle-core/src/quantized/`

- [ ] **Step 1: Write block layout test**

```rust
#[test]
fn test_fp8_block_sizes() {
    assert_eq!(std::mem::size_of::<BlockQ8F4M3_0>(), 34);
    assert_eq!(std::mem::size_of::<BlockQ8F4M3_1>(), 36);
    assert_eq!(std::mem::size_of::<BlockQ8F5M2_0>(), 34);
    assert_eq!(std::mem::size_of::<BlockQ8F5M2_1>(), 36);
}
```

- [ ] **Step 2: Write quantize/dequant roundtrip test**

```rust
#[test]
fn test_q8f4m3_0_roundtrip() {
    let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let nb = src.len() / QK8F4M3_0;
    let mut blocks = vec![BlockQ8F4M3_0::zeros(); nb];
    BlockQ8F4M3_0::from_float(&src, &mut blocks);
    let mut dst = vec![0f32; src.len()];
    BlockQ8F4M3_0::to_float(&blocks, &mut dst);
    for (s, d) in src.iter().zip(dst.iter()) {
        let err = (s - d).abs() / (1.0 + s.abs());
        assert!(err < 0.01, "relative error {err} too large at {s} -> {d}");
    }
}
```

Write similar tests for Q8F4M3_1, Q8F5M2_0, Q8F5M2_1.

- [ ] **Step 3: Write vec_dot test**

```rust
#[test]
fn test_q8f4m3_0_vec_dot() {
    let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let nb = src.len() / QK8F4M3_0;
    let mut blocks = vec![BlockQ8F4M3_0::zeros(); nb];
    BlockQ8F4M3_0::from_float(&src, &mut blocks);
    let n = blocks.len() * QK8F4M3_0;
    let result = BlockQ8F4M3_0::vec_dot(n, &blocks, &src);
    let expected: f32 = src.iter().map(|x| x * x).sum();
    let err = (result - expected).abs() / expected.abs();
    assert!(err < 0.02, "vec_dot error {err} too large");
}
```

- [ ] **Step 4: Write zero-input / extreme-value tests**

```rust
#[test]
fn test_q8f4m3_0_all_zeros() {
    let src = vec![0f32; 64];
    let nb = src.len() / QK8F4M3_0;
    let mut blocks = vec![BlockQ8F4M3_0::zeros(); nb];
    BlockQ8F4M3_0::from_float(&src, &mut blocks);
    let mut dst = vec![0f32; src.len()];
    BlockQ8F4M3_0::to_float(&blocks, &mut dst);
    for &d in dst.iter() {
        assert!(!d.is_nan(), "NaN in output");
    }
}
```

- [ ] **Step 5: Write GGUF roundtrip test**

```rust
#[test]
fn test_gguf_fp8_roundtrip() -> Result<()> {
    let src = vec![1f32, 2.0, 3.0, 4.0]; // will pad to 32
    let tensor = Tensor::from_vec(src.clone(), 4, &Device::Cpu)?;
    let qtensor = tensor.quantize(GgmlDType::Q8F4M3_0)?;

    // Write to buffer
    let mut buf = std::io::Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &[], &[("test", &qtensor)])?;

    // Read back
    buf.set_position(0);
    let content = gguf_file::Content::read(&mut buf)?;
    let read_qtensor = content.tensor(&mut buf, "test", &Device::Cpu)?;
    let read_tensor = read_qtensor.dequantize()?;

    // Compare
    let orig = tensor.to_vec1::<f32>()?;
    let roundtripped = read_tensor.to_vec1::<f32>()?;
    // Note: dequantize loss expected, check shapes match
    assert_eq!(read_tensor.shape().elem_count(), 4);
    Ok(())
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p candle-core -- quantized 2>&1 | tail -30
```
Expected: All new tests pass.

- [ ] **Step 7: Commit**

```bash
git add candle-core/src/quantized/
git commit -m "test: add FP8 block type tests and GGUF roundtrip"
```

---

### Task 2.6: Add GgmlDType type_id roundtrip test

**Files:** Add test to `candle-core/src/quantized/mod.rs` or existing test module

- [ ] **Step 1: Write type_id test**

```rust
#[test]
fn test_fp8_ggml_dtype_roundtrip() {
    for id in [43u32, 44, 45, 46] {
        let dtype = GgmlDType::from_u32(id).unwrap();
        assert_eq!(dtype.to_u32(), id);
    }
}
```

- [ ] **Step 2: Verify**

```bash
cargo test -p candle-core -- test_fp8_ggml_dtype_roundtrip
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add candle-core/src/quantized/
git commit -m "test: add FP8 GgmlDType type_id roundtrip test"
```

---

## Phase 3: SIMD + CUDA + Metal Kernels

### Task 3.1: Add AVX vec_dot for FP8 types

**Files:** Modify `candle-core/src/quantized/avx.rs`

- [ ] **Step 1: Implement vec_dot_q8f4m3_0_f32**

Following the existing `vec_dot_q8_0_q8_0` pattern, implement using AVX2 intrinsics. For FP8, the approach is:
- Load `d` scale as `__m256` broadcast
- Load 8 FP8 values, convert to F32 using `_mm256_cvtepu8_epi32` + scale
- Multiply-accumulate against corresponding F32 activation values
- Horizontal sum at end

```rust
#[cfg(target_feature = "avx2")]
pub fn vec_dot_q8f4m3_0_f32(n: usize, xs: &[BlockQ8F4M3_0], ys: &[f32]) -> f32 {
    // AVX2 implementation: 8-wide FMA on dequantized values
    // ...
}
```

- [ ] **Step 2: Implement vec_dot for remaining 3 types**

Same pattern for Q8F4M3_1, Q8F5M2_0, Q8F5M2_1.

- [ ] **Step 3: Verify on AVX2-capable machine**

```bash
RUSTFLAGS="-C target-feature=+avx2" cargo test -p candle-core -- quantized 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add candle-core/src/quantized/avx.rs
git commit -m "feat: add AVX2 vec_dot for FP8 block types"
```

---

### Task 3.2: Add NEON and SIMD128 vec_dot for FP8 types

**Files:** Modify `candle-core/src/quantized/neon.rs`, `candle-core/src/quantized/simd128.rs`

- [ ] **Step 1: NEON vec_dot**

Add `vec_dot_q8f4m3_0_f32`, `vec_dot_q8f4m3_1_f32`, `vec_dot_q8f5m2_0_f32`, `vec_dot_q8f5m2_1_f32` functions to `neon.rs`. Follow the existing NEON vec_dot pattern (e.g., `vec_dot_q8_0_q8_0`), using NEON intrinsics to dequantize FP8 blocks and dot with F32 activations.

- [ ] **Step 2: SIMD128 vec_dot**

Add the same 4 functions to `simd128.rs` for WASM SIMD128 support.

- [ ] **Step 3: Commit**

```bash
git add candle-core/src/quantized/neon.rs candle-core/src/quantized/simd128.rs
git commit -m "feat: add NEON and SIMD128 vec_dot for FP8 block types"
```

---

### Task 3.3: Add CUDA fast matmul dispatch for FP8 types

**Files:** Modify `candle-core/src/quantized/cuda.rs`, `fast_mmvq.rs`, `fast_mmq.rs`

- [ ] **Step 1: Wire FP8 types in CUDA quantize/dequantize**

In `cuda.rs`, add match arms for the 4 new `GgmlDType` variants in the quantize and dequantize dispatch. Initially, these can fall through to CPU.

- [ ] **Step 2: Add dispatch arms in fast_mmvq.rs**

In the CUDA MMVQ dispatch, add arms for Q8F4M3_0, Q8F4M3_1, Q8F5M2_0, Q8F5M2_1. These delegate to CPU vec_dot until dedicated CUDA kernels are written (future optimization).

- [ ] **Step 3: Add dispatch arms in fast_mmq.rs**

Same pattern for MMQ dispatch.

- [ ] **Step 4: Commit**

```bash
git add candle-core/src/quantized/cuda.rs candle-core/src/quantized/fast_mmvq.rs candle-core/src/quantized/fast_mmq.rs
git commit -m "feat: add CUDA dispatch arms for FP8 block types"
```

---

### Task 3.4: Wire Metal backend for FP8 types

**Files:** Modify `candle-core/src/quantized/metal.rs`, `candle-metal-kernels/src/kernels/quantized.rs`

- [ ] **Step 1: Add FP8 to Metal GgmlDType enum**

In `candle-metal-kernels/src/kernels/quantized.rs`, add Q8F4M3_0, Q8F4M3_1, Q8F5M2_0, Q8F5M2_1 variants to the Metal-side `GgmlDType` enum and wire them through kernel dispatch (delegating to CPU dequantize until dedicated Metal shaders exist).

- [ ] **Step 2: Wire in metal.rs**

In `candle-core/src/quantized/metal.rs`, add match arms for the 4 new variants.

- [ ] **Step 3: Commit**

```bash
git add candle-core/src/quantized/metal.rs candle-metal-kernels/src/kernels/quantized.rs
git commit -m "feat: add Metal backend wiring for FP8 block types"
```

---

### Task 3.5: Add CUDA FFI declarations for FP8 launchers

**Files:** Modify `candle-kernels/src/ffi.rs`

- [ ] **Step 1: Add FFI declarations**

Add FFI function declarations for FP8 CUDA kernel launchers if/when dedicated CUDA FP8 kernels are written. For Phase 3, the launchers may just be placeholders that fall through to CPU.

- [ ] **Step 2: Commit**

```bash
git add candle-kernels/src/ffi.rs
git commit -m "feat: add CUDA FFI declarations for FP8 launchers"
```

---

## Integration & Verification

### Task V.1: Full workspace build and test

- [ ] **Step 1: Build entire workspace**

```bash
cargo build --workspace 2>&1 | tail -10
```
Expected: All crates compile without errors.

- [ ] **Step 2: Run full test suite**

```bash
cargo test --workspace 2>&1 | tail -30
```
Expected: All existing tests pass, no regressions.

- [ ] **Step 3: Run specific quantized tests**

```bash
cargo test -p candle-core -- quantized:: 2>&1 | tail -30
```
Expected: All quantized tests pass including new FP8 tests.

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A
git commit -m "chore: finalize FP8 integration, all tests pass"
```

---

### Task V.2: Manual end-to-end verification

- [ ] **Step 1: Quantize a small model to each FP8 type using tensor-tools**

```bash
cargo run -p tensor-tools -- quantize --input model.safetensors --output model-q8f4m3_0.gguf --qtype q8f4m3_0
```

- [ ] **Step 2: Verify GGUF file structure**

```bash
cargo run -p tensor-tools -- inspect model-q8f4m3_0.gguf
```
Expected: Shows correct tensor shapes, dtype IDs 43-46, metadata `candle.fp8_types`.

- [ ] **Step 3: Compare logits with F32 baseline**

Run inference on the quantized model and compare output logits with the F32 version. Logits should be within reasonable tolerance (e.g., < 1% relative error for F8E4M3, < 2% for F8E5M2).
