# dbuf Autotune Analysis — What Wins and Why

**Data**: 1360 configs, full sweep CSV [`dbuf-sweep-an-h100-node-26.csv`](dbuf-sweep-an-h100-node-26.csv)
(a-h100-cluster H100, 2026-09-26, N=4096 FP32, warmup 3 + 100-iter event timing per config)

This note answers: which dimensions decide performance, and what the causal
mechanism is. All numbers below are computed from the sweep CSV.

## The winner and its neighborhood

Winner: **BM=128 BN=256 BK=8 WM=64 WN=64 WNITER=2 TM=8 TN=4 NT=256 → 37.60T**
(72.4% of cuBLAS FP32). The entire top-10 is one family: BM=128, BN=256,
NT=256, BK∈{8,16}, with warp geometry (WM/WN = 64/64, 32/128, 128/32) varying
within 0.7T. Once the block shape and thread count are fixed, how you cut the
warp tile inside barely matters.

## Per-dimension medians and maxima (from the sweep)

| Dim | Value: median / max (n) |
|---|---|
| BM | 64: 28.9/36.4T (330) · **128: 26.0/37.6T (522)** · 256: 6.6/34.5T (508) |
| BN | 64: 24.3/31.5T (330) · 128: 26.0/34.6T (522) · **256: 6.6/37.6T (508)** |
| BK | **8: 28.9/37.6T (452)** · 16: 25.5/37.3T (454) · 32: 18.8/33.3T (454) |
| NT | 32: 5.7/29.8T · 64: 7.8/34.2T · 128: 25.5/36.4T · **256: 30.7/37.6T** · 512: 25.4/35.1T · 1024: 4.3/32.5T |
| accumulators/thread | 32: 25.7/32.5T · 64: 28.8/35.1T · **128: 29.8/37.6T** · 256: 6.1/8.8T · 512: 1.3/1.9T |

Best-to-worst config spread: 28× (37.60T vs 1.33T).

## The causal story (four mechanisms, all visible in the data)

**1. NT=256 is the strongest single predictor.** NT=32/64 configs median
5.7-7.8T: with so few threads per block, each thread carries a huge serial
workload and there are not enough warps for the cp.async pipeline to overlap
with. NT=1024 median 4.3T: the SM can host at most 2 such blocks (2048-thread
limit), one register-pressure hiccup and occupancy collapses to 1 block; tail
quantization is also worst. NT=256 sits at the sweet spot: 8 warps/block,
2-3 resident blocks/SM, 16-24 schedulable warps.

**2. Accumulator count is the register-pressure cliff.** acc=128
(WMITER·TM × WNITER·TN per thread) is the sweet spot: 128 independent FMA
chains per thread. acc=256 or 512 → medians 6.1T/1.3T — the accumulator array
no longer fits the register file; spills go through local memory (L1/L2) and
performance falls off a cliff. Note the asymmetry: acc=128 is *good* precisely
because 128 independent accumulators give massive ILP — instruction-level
parallelism substitutes for occupancy (the resident-block count is
register-limited to ~2 blocks/SM at ~190 regs/thread).

**3. Block-tile arithmetic intensity sets the ceiling; BN=256 wins the trade.**
AI = 2·BM·BN / ((BM+BN)·4) FLOP per GMEM byte (independent of BK — see #4).
Winner (128×256): AI=42.7. Old default (128×128): AI=32.0. Going wider to
256×256 gives AI=64 but forces acc≥256 or NT≤64 in most valid tilings —
which lands on the register/spill cliff (mechanism #2). 128×256 is the
highest AI you can reach while keeping acc=128 and NT=256.

**4. BK is a pure pipeline-granularity knob (NOT reuse).** AI is
BK-independent (the BK terms cancel), so BK trades SMEM per stage against
pipeline refresh rate. BK=8 at BN=256 → 24KB SMEM/block → SMEM allows 8
blocks/SM; BK=16 → 48KB → 4 blocks; BK=32 → 96KB → 2 blocks. The measured
ordering (8 > 16 > 32, medians 28.9/25.5/18.8T) matches: with double
buffering, shallower K-chunks mean the prefetch pipeline switches buffers
twice as often, and more resident blocks give the scheduler more overlap
opportunities. Reuse is not sacrificed because block-level AI never changed.

**Why 256×BM/BN medians are catastrophic (6.6T) yet max is 34.5T**: the
median is dragged down by the (BM or BN)=256 configs that are only valid with
few threads or huge accumulators (mechanisms #1/#2); the few 256-side configs
that keep acc=128 + NT=256 (e.g. BM=256 BN=128 with 8 warps of 32×64) do
reach 34.5T. The dimension label alone is not the cause — the (NT, acc)
pair is.

## The TM/TN mirror asymmetry, again

TM=8/TN=4 reaches 37.60T; the mirrored TN=8 variant caps at 36.35T. Same
effect as sgemm-ladder-h100.md lesson #4 (TM/TN swap cost −30% there): the
inner-N loop's regN lifetime and the SMEM read pattern favor the tall-thread
tile. autotune rediscovers it without being told.

## Practical takeaway for the next rungs

- Autotune space should include NT and derived acc as FIRST-CLASS dims —
  they explain more variance than WM/WN geometry.
- The winner family (BM=128, BN=256, NT=256, BK=8) is the natural starting
  point for WGMMA porting: same block tile, tensor-core fragments replace the
  thread tile.
- Sweep cost: 1360 configs × ~1.9s ≈ 43 min single-GPU — cheap enough to
  re-run per architecture; the H100 optimum (BN=256, BK=8) differs from the
  A100-era defaults (128,128,8), confirming autotune.md's "each new dimension
  unlocks new trade-offs".
