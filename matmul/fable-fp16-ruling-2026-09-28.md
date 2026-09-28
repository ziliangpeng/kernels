# Fable ruling: why FP16 ≈ FP32 on Hopper CUDA cores (2026-09-28)

Relayed via Herald (#req-20260928-001, f5m via midagent, API 190s). Full text
at /tmp/herald_fp16arch_ruling.txt on maxbot.

## Core thesis — the binding constraint is ISSUE SLOTS, not the cvt pipe

Hopper: 4 SMSP/SM, 1 warp instruction issued per SMSP per clock → 4×32 = 128
FFMA/clk/SM = EXACTLY the FP32 peak. FP32 peak requires 100% of issue slots
to be FFMA; every extra instruction (LDS/address math/cvt) steals an FFMA
slot directly. cvt:FMA 0.25-0.31 → 1/1.28 ≈ 0.78; add back byte savings →
0.83-0.91x observed. Fits without any "1/8 pipe" assumption.

## Where our earlier story was wrong

1. "cvt pipe 1/8 throughput" is FALSIFIED by our own data: at cvt:FMA=0.28 a
   16/clk/SM cvt pipe would cost 2.24 FMA-times → FP16 warptile would be
   2-3x slower, not 0.83x. Fable judges ptxas compiles __half2float to
   HADD2.F32 (FP16 pipe, full speed, 1 issue/element) — the tax is 1 issue
   slot per element, not 8.
2. "Deeper reuse = higher cvt/FMA" was backwards: ratio = (TM+TN)/(TM·TN)
   falls with tile size; what actually happens is deeper reuse → kernel
   near issue saturation → non-FFMA instructions become pure loss, byte
   savings have nowhere to cash in. Tax didn't grow; the slack vanished.
3. This fully explains the ordering flip: warptile pushes FP32 to 59% of
   peak (issue-saturated), FP16's cvt steals slots there; vectorized has
   slack — halved load instructions cash in, cvt hides in the slack.
4. HFMA2 = 2x FP32 FLOPs (256 vs 128 results/clk/SM, whitepaper 133.8/66.9
   consistent with our 53c14dd) — but the 2x only exists with FP16
   accumulation. No scalar half×half→fp32-acc instruction exists; the only
   mixed-precision FMA is HMMA (Tensor Core).

## Recommended path (His)

- At most two more scalar rungs, both pedagogical:
  (i) SMEM-fill cvt — convert during GMEM→SMEM fill (SMEM holds fp32); tax
      drops from O(N³(1/TM+1/TN)) to O(N³(1/BM+1/BN)) ≈ 1.5% at 128×128;
      inner loop identical to FP32; GMEM byte savings kept; expect FP16's
      FIRST compute-bound win (+2-5% over FP32). Cheapest model-falsification
      experiment for the "per-element cvt tax" theory.
  (ii) HFMA2 chunked accumulation — the only scalar path past the FP32
      compute ceiling (66.9→133.8T); fp16 partial sums 16-32 steps then
      HADD2.F32+FADD flush; expect 1.5-1.8x FP32; needs relaxed verify.
- Then mma.sync (m16n8k16 HMMA, ldmatrix, sm_80 style — the FlashAttention-2
  generation skeleton) as ONE intermediate step, THEN WGMMA+TMA+warp-spec.
- Teaching point: ideal scalar HFMA2 ~100T is still 7x below cuBLAS FP16 TC
  728.7T — the strongest "why Tensor Cores" slide.

## Verify list He left

(1) SASS: is inner-loop cvt HADD2.F32 or F2F; LDS per FFMA FP16 vs FP32.
(2) Microbench: pure FFMA vs +22% __half2float linearity; HFMA2 issue-bound
    2x FLOPs check. (3) Whitepaper 133.8/66.9 + CUDA guide 256/128. (4) His
    GA100 HFMA2.MMA memory.

## Open decision (Ziliang)

1) SMEM-fill cvt only, then mma.sync. 2) Both scalar rungs. 3) Straight to
mma.sync.
