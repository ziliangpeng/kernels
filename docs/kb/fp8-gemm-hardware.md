# FP8 GEMM Hardware Pipeline

## Can FP8 GEMM run on CUDA cores?

**No.** H100 CUDA cores have no FP8 ALU. `cuda_fp8.h` provides only conversion functions (`__nv_cvt_fp8_to_halfraw`, etc.) — zero FP8 arithmetic intrinsics.

FP8 on CUDA cores = emulation: convert FP8→FP16, do FP16 multiply, convert back. This is **slower than FP16 GEMM** because of conversion overhead with no compute benefit.

Meta's PyTorch Conference talk (Luca Wehrstedt, FAIR): "The only computation that gets faster with fp8 is the one that goes through TensorCores."

FP8's 2× speedup comes **entirely from Tensor Core**. Without Tensor Core, FP8 has no compute advantage.

## Memory path: FP8 stays 8-bit end-to-end

FP8 input can remain 8-bit from HBM all the way to the Tensor Core register:

```
HBM:        FP8 (1 byte/element, packed 2 per byte lane)
  ↓ load (8-bit load — saves 2× bandwidth vs FP16)
L2 cache:   FP8 (cached as-is)
  ↓
L1/shared:  FP8 (1 byte/element — fits 2× more tiles than BF16)
  ↓ load to register
Register:   FP8 (packed — 4 FP8 per 32-bit register)
  ↓ feed to Tensor Core
Tensor Core: native FP8 MMA → FP32 accumulator
```

**FP8 is NOT expanded to 16-bit before reaching compute.** The Tensor Core directly receives packed FP8 operands. This is hardware-native, not emulation.

## What Tensor Core does internally

H100 Tensor Core has native FP8 MMA instruction (`mma.sync` with FP8 element type). Internal flow:

```
Input:  packed FP8 E4M3 (4 elements per 32-bit register)
        ↓ hardware decode (4-bit exponent + 3-bit mantissa)
Multiply: FP8 × FP8 → internal high precision (≥FP16, sufficient for 6-bit mantissa product)
        ↓
Accumulate: FP32 accumulator (same as BF16 GEMM)
        ↓
Output: FP32 accumulator register
```

This is NOT "expand to FP16 then do FP16 multiply." The hardware directly decodes the FP8 format and does a native multiply into the FP32 accumulator.

### Accumulator precision nuance (Sol review — deeper than "FP32")

"FP32 accumulator" describes the API/architectural contract, not the silicon. Three distinct things:

1. **Exposed accumulator type** (what `mma.sync` says: `.f32`)
2. **Internal partial accumulator precision** — Hopper FP8 native accumulation uses an internal precision *wider than FP16 but narrower than FP32*
3. **Promotion cadence** — non-fast-accum mode periodically promotes partials to full precision; fast-accum mode skips the promotion (faster, less accurate)

PTX also has FP8 MMA variants with **FP16 accumulator** (`mma...f16.e4m3.e5m2.f16`). So the full space:

| Mode | Accumulator | Notes |
|---|---|---|
| fast accumulation (ON) | narrower internal partials, no periodic promotion | fastest, less accurate |
| fast accumulation (OFF) | periodic promotion to higher precision | default |
| FP16 accumulator MMA | FP16 | exists in PTX ISA |

When analyzing FP8 numerics, don't just ask "FP16 or FP32 accumulator?" — also ask: is fast-accum on? What's the promotion interval? What precision does split-K reduction use? Is rounding per-product or per-group?

Sources: NVIDIA PTX ISA, CUDA Tile IR stability docs (https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html), Hopper Architecture whitepaper, Meta PyTorch Conference talk, Sol review.

H100 FP8 Tensor Core tile: 16×16×32 (K dimension = 32, double BF16's 16, because FP8 elements are half the size — same register holds 2× more).

## FP8 GEMM throughput (H100)

| Metric | BF16 | FP8 | Speedup |
|---|---|---|---|
| Tensor Core TFLOPS | 989 | 1979 | 2× |
| Bytes per element | 2 | 1 | 2× memory |
| K-dimension tile | 16 | 32 | 2× per MMA |

Three factors compound: 2× compute + 2× memory bandwidth + 2× K-tile → overall ~2× speedup.

## E4M3 vs E5M2 in GEMM

| Format | Exponent | Mantissa | Max value | Use case |
|---|---|---|---|---|
| E4M3 | 4 bit | 3 bit | ±448 | Forward pass (weights, activations — bounded values, need precision) |
| E5M2 | 5 bit | 2 bit | ±57344 | Backward pass (gradients — large range, less precision needed) |

Forward GEMM uses E4M3. Backward GEMM (weight gradient, activation gradient) can use E5M2.

## When FP8 without Tensor Core makes sense

| Scenario | Why |
|---|---|
| Memory-bound GEMV (decode M=1) | FP8 weight saves 2× HBM bandwidth, even with FP16 compute |
| Fused dequant + elementwise | Load FP8 → apply scale → immediately compute as FP16, avoid materializing FP16 tensor |
| Elementwise / reduction / norm | Tensor Core not suitable for these ops; FP8 as compressed input |
| Old GPU without FP8 Tensor Core | FP8 as storage/compression format for bandwidth-bound workloads |
| Large dense GEMM on H100 | **No reason to use CUDA core** — should always use Tensor Core |

Sources: NVIDIA PTX ISA, Hopper Architecture whitepaper, Meta PyTorch Conference talk, Sol review.
