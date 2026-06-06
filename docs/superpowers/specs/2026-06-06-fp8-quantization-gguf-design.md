# FP8 Quantization & GGUF Model Format Support — Design Spec

**Date:** 2026-06-06
**Status:** Design approved, pending implementation plan

---

## Context

Candle already supports `F8E4M3` as a full `DType` (CPU + CUDA) and has a mature GGUF quantized tensor system with integer-based block types (Q4_0 through Q8K). However, there is no FP8-based block quantization in the `GgmlDType` path, and `F8E5M2` does not exist anywhere in the codebase. This spec covers adding both FP8 formats as first-class quantization types with full backend support.

---

## Design Summary

Three sequential phases, with Phase 1 split into two parallel sub-tracks.

### Phase 1a: F8E5M2 DType

Add `F8E5M2` as a real `DType` alongside the existing `F8E4M3`. Mechanical work following existing `F8E4M3` patterns.

**Files:**
- `candle-core/src/dtype.rs` — `DType::F8E5M2` variant, `WithDType`, `FloatDType` impls
- `candle-core/src/scalar.rs` — `Scalar::F8E5M2` variant
- `candle-core/src/cpu_backend/mod.rs` — `CpuStorage::F8E5M2`, conversion impls
- `candle-core/src/cuda_backend/mod.rs` — `CudaStorageSlice::F8E5M2`, `cuda_dtype!`
- `candle-core/src/metal_backend/mod.rs` — `MetalStorage::F8E5M2` variant
- `candle-core/src/safetensors.rs` — `DType::F8E5M2` ↔ `st::Dtype::F8_E5M2`
- `candle-core/src/display.rs` — Display impl
- `candle-core/src/cpu/kernels.rs` — `VecOps` for `f8e5m2`
- `candle-kernels/src/fill.cu` — `fill_f8_e5m2`, `copy2d_f8_e5m2`
- `candle-kernels/src/cast.cu` — `cast_f8_e5m2`
- `candle-kernels/src/affine.cu` — `affine_f8_e5m2`
- `candle-kernels/src/ternary.cu` — `where_*_f8_e5m2`
- `candle-metal-kernels/src/` — Metal stubs for F8E5M2

### Phase 1b: GGUF Block Type Design (parallel track)

Design artifact produced while Phase 1a is implemented.

#### New GgmlDType Variants

| Variant | FP8 Format | Style | GGUF ID | Block Size |
|---------|-----------|-------|---------|------------|
| `Q8F4M3_0` | F8E4M3 | Symmetric (Q8_0-style) | 43 | 34 bytes |
| `Q8F4M3_1` | F8E4M3 | Asymmetric (Q8_1-style) | 44 | 36 bytes |
| `Q8F5M2_0` | F8E5M2 | Symmetric (Q8_0-style) | 45 | 34 bytes |
| `Q8F5M2_1` | F8E5M2 | Asymmetric (Q8_1-style) | 46 | 36 bytes |

IDs 43-46 are chosen to avoid conflicts with upstream ggml (max ID currently 42). Documented in GGUF metadata under `candle.fp8_types`.

#### Block Structures (`#[repr(C)]`)

Symmetric (34 bytes):
```rust
struct BlockQ8F4M3_0 { d: f16, qs: [F8E4M3; 32] }
struct BlockQ8F5M2_0 { d: f16, qs: [F8E5M2; 32] }
```

Asymmetric (36 bytes):
```rust
struct BlockQ8F4M3_1 { d: f16, m: f16, qs: [F8E4M3; 32] }
struct BlockQ8F5M2_1 { d: f16, m: f16, qs: [F8E5M2; 32] }
```

#### Dequantization

- Symmetric: `value[i] = qs[i].to_f32() * d.to_f32()`
- Asymmetric: `value[i] = qs[i].to_f32() * d.to_f32() + m.to_f32()`

#### Quantization (from_float)

- Symmetric: `d = max(|values|) / max_fp8_value`, `qs[i] = round(clamp(values[i] / d))`
- Asymmetric: `m = min(values)`, `d = (max - min) / max_fp8_value`, `qs[i] = round(clamp((values[i] - m) / d))`

### Phase 2: GGUF FP8 Block Types (CPU Implementation)

Make the block types real on CPU — quantize, dequantize, vec_dot, GGUF serialization.

**Files:**
- `candle-core/src/quantized/mod.rs` — 4 new `GgmlDType` variants, `from_u32`/`to_u32`, `type_size`/`block_size`, `QStorage::from_data` arms
- `candle-core/src/quantized/k_quants.rs` — 4 block structs + `GgmlType` trait impls (scalar CPU: `VecDotType = f32`)
- `candle-core/src/quantized/gguf_file.rs` — Writer arms, reader type ID recognition
- `candle-core/src/quantized/ggml_file.rs` — `qtensor_from_ggml` arms for IDs 43-46
- GGUF metadata key `candle.fp8_types` written on export

### Phase 3: SIMD + CUDA + Metal Kernels

Performance — SIMD vec_dot for AVX/NEON/WASM, CUDA fast matmul dispatch, Metal kernel support.

**Files:**
- `candle-core/src/quantized/avx.rs` — `vec_dot` for each FP8 type
- `candle-core/src/quantized/neon.rs` — Same for NEON
- `candle-core/src/quantized/simd128.rs` — Same for WASM SIMD128
- `candle-core/src/quantized/cuda.rs` — CUDA quantize/dequantize wiring
- `candle-core/src/quantized/fast_mmvq.rs` — Dispatch arms
- `candle-core/src/quantized/fast_mmq.rs` — Dispatch arms
- `candle-core/src/quantized/metal.rs` — Metal backend wiring
- `candle-metal-kernels/src/kernels/quantized.rs` — FP8 in Metal `GgmlDType` enum
- `candle-kernels/src/ffi.rs` — FP8 CUDA launcher FFI declarations

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Block size mismatch | Pad last block with zeros (matches Q8_0) |
| All-zero / all-identical block | Clamp `d` to `f16::MIN_POSITIVE` to avoid NaN |
| FP8 overflow | `float8` crate saturates; additional clamp before conversion |
| Corrupt block data (NaN bit patterns) | Propagate NaN, no panic |
| Unknown FP8 type ID in reader | `Err("unsupported GGUF quant type {id}, expected 43-46")` |
| CUDA/Metal op before kernels exist | Fall through to CPU dequantize → compute → requantize |

---

## Testing

### Unit Tests
- Block layout: `#[repr(C)]` sizes, alignment
- Quantize/dequant roundtrip: max error < 0.5% (F8E4M3), < 1% (F8E5M2)
- `vec_dot` vs F32 reference within epsilon
- Zero inputs, constant inputs, extreme values — no panics, no NaN
- `to_float` / `from_float` inverses
- `GgmlDType::type_size()` / `block_size()` correctness
- GGUF type ID roundtrip: `from_u32(id).to_u32() == id`
- F8E5M2 DType conversions, scalar ops, display
- `CpuStorage::F8E5M2` / `CudaStorage::F8E5M2` allocation

### Integration Tests
- GGUF write → read roundtrip for all 4 FP8 quant types
- GGUF write → read for F8E5M2 DType path
- Small model quantize → forward pass → logits within tolerance of F32 baseline

### Property-Based Tests
- `from_float` + `to_float` never panics for any `Vec<f32>` (any length, including empty)
- `vec_dot` is approximately commutative

### SIMD Tests (Phase 3)
- AVX/NEON/SIMD128 `vec_dot` matches scalar reference within 1 ULP

---

## Verification

End-to-end validation:
1. `cargo build --workspace` — all crates compile
2. `cargo test --workspace` — all existing tests pass, new tests pass
3. `cargo test --workspace --features cuda` — CUDA tests pass (if GPU available)
4. `cargo test --workspace --features metal` — Metal tests pass (if macOS)
5. Manual: quantize a small model to each FP8 type, run inference, compare logits to F32
6. Manual: verify GGUF files written by Candle are readable by `tensor-tools`
