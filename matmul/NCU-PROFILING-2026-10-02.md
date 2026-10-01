# NCU full-ladder profiling (2026-10-02, kdev-profiling pod, H100 SXM)

Environment: kdev-profiling Deployment (ziliang-xp ns, cuda:12.6.3-devel, privileged, 1xH100),
ncu 2024.3.2 (clock-control base, SM locked 1.43 GHz; memory clocks untouched), 137 artifacts
in PVC /workspace/profiling/ (24 variants SoL+Occupancy @8192; 9 variants x 4 sections x
{4096,8192}; 5 nsys gpu-metrics runs; all text-ified).

## Pass 1 — all variants @8192 (first kernel instance, locked clocks)

| method | dur ms | tensor% | sm% | mem% | L2% | occ% | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| wmma | 48.12 | 4.4 | 17.5 | 98.9 | 19.4 | 74.4 | pure DRAM-bound |
| wgmma_v2 | 11.08 | 12.8 | 12.9 | 96.5 | 79.5 | 49.4 | mem-bound |
| wgmma_v3 | 11.78 | 12.0 | 12.2 | 91.7 | 57.4 | 49.5 | mem-bound |
| wgmma_v4 | 7.65 | 18.3 | 21.0 | 84.4 | 51.5 | 49.3 | mem-bound |
| wgmma_v5 | 8.24 | 17.1 | 23.6 | 82.9 | 50.5 | 48.8 | mem-bound |
| wgmma_v5_1 | 7.53 | 18.6 | 25.7 | 90.6 | 62.1 | 49.7 | mem-bound |
| wgmma_v6 | 6.18 | 22.9 | 22.9 | 91.5 | 71.9 | 24.7 | mem-bound |
| wgmma_v7 | 6.07 | 23.3 | 23.3 | 85.9 | 53.4 | 12.5 | mem-bound |
| wgmma_v8 | 6.69 | 20.9 | 20.9 | 82.4 | 47.8 | 18.5 | mem-bound |
| wgmma_v8_1 | 6.84 | 20.5 | 20.5 | 80.6 | 45.6 | 18.5 | mem-bound |
| wgmma_v9 | 2.12 | 67.1 | 67.1 | 60.5 | 71.4 | 13.8 | turning point (TMA) |
| wgmma_v9_1 | 2.98 | 47.4 | 47.4 | 60.1 | 74.5 | 13.9 | slower v9 |
| wgmma_v9_2 | 2.10 | 67.3 | 67.3 | 61.8 | 71.7 | 13.8 | = v9 |
| wgmma_v9_3 | 2.03 | 69.8 | 69.8 | 52.3 | 54.1 | 13.8 | |
| wgmma_v9_4 | 2.08 | 68.1 | 68.1 | 46.8 | 63.1 | 13.8 | negative variant |
| wgmma_v9_5 | 2.00 | 70.9 | 70.9 | 51.7 | 51.8 | 18.3 | TMA epilogue |
| wgmma_v9_6 | 1.72 | 82.9 | 82.9 | 64.6 | 64.6 | 13.7 | 4-stage |
| wgmma_v9_7 | 1.62 | 88.5 | 88.5 | 58.6 | 58.6 | 18.2 | tensor-pipe-bound |
| wgmma_v9_8 | 1.64 | 86.2 | 86.2 | 56.6 | 54.8 | 18.8 | tensor-pipe-bound |
| wgmma_v9_9 | 1.59 | 88.6 | 88.6 | 60.8 | 60.2 | 18.2 | tensor-pipe-bound |
| wgmma_v9_10 | 2.19 | 64.7 | 64.7 | 42.5 | 37.2 | 14.1 | 256x256 fail (2-stage) |
| wgmma_v9_11 | 1.60 | 89.1 | 89.1 | 66.6 | 67.8 | 18.2 | tensor-pipe-bound (champ) |
| wgmma_v9_12 | 2.10 | 67.9 | 67.9 | 46.4 | 46.6 | 18.3 | multicast flat (impl tax) |
| wgmma_v9_13 | 1.65 | 86.2 | 86.2 | 59.1 | 59.2 | 18.2 | multicast fixed, still flat |

Counter-backed conclusions:
1. wmma/v2..v8_1 are MEMORY-bound (mem% 80-99%) — the whole pre-TMA ladder is bandwidth-limited;
   tensor pipe never above 24%.
2. v9 (TMA) is the phase transition: tensor 67% / mem 60% — TMA offloads the memory path.
3. v9_7/v9_9/v9_11 top tier: tensor pipe 88.5-89.1% = hardware-limited; the design point is
   saturated. Remaining ~11% = issue gaps (see Pass 2 stalls).
4. Multicast v9_12/13 confirm Fable's mechanism: at 128x256 L2% is only 46-59% (not the
   bottleneck), so saving L2 reads buys nothing — the disease multicast treats doesn't exist
   in this geometry.
5. Occupancy story: pre-TMA kernels 49% occ (2 CTAs/SM) vs TMA kernels 13-18% (1 CTA/SM,
   SMEM-limited) — low occupancy does NOT prevent tensor saturation on Hopper (async WGMMA
   depth substitutes for warp count).

## Pass 2 — deep (WarpState stalls top-5 + L1/L2 hit), per kernel

- wgmma_v6 (pre-TMA): long_scoreboard 17.7 (waiting on SMEM/L2 loads) >> barrier 3.2 — classic
  latency-bound; L2 hit 85.6%, L1 hit 2.6%.
- wgmma_v9_3 (TMA + 2-stage): barrier 5.4, long_scoreboard 4.2, lg_throttle 2.8; L1 hit 76%.
- wgmma_v9_6 (4-stage): barrier 6.4, long_scoreboard 1.1 (pipeline depth fixed the scoreboard
  stalls); L1 hit 72%.
- wgmma_v9_7/9_9/9_11 (top tier): barrier 9.3-11.3 DOMINANT, long_scoreboard ~1 — the kernel
  is mbarrier-paced; tensor pipe does 89% but warps spend their stall budget on barrier waits
  between wgmma groups. L2 hit 71-77%.
- wgmma_v9_10: L1 hit 78.7% but tensor only 64.7% — the 256x256 2-stage pipeline cannot hide
  latency (barrier 4.5 + wait 3.0 + long_scoreboard 2.8 spread).
- wgmma_v9_12/13 (multicast): barrier ~9.6-9.8 + long_scoreboard 3.1-3.3 (v9_12) vs 0.9
  (v9_13) — v9_12's split-issue serialization shows up exactly as predicted.

Note: L1 hit 0.0% for v9_7/9_9/9_11/9_12/9_13 is expected — TMA writes SMEM via the async
proxy, bypassing L1; the "L1" traffic that exists is the wgmma SMEM reads which count in a
different counter. v9_3/9_6/9_10 show L1 hits because their epilogues read SMEM through L1.

## Pass 3 — nsys gpu-metrics (free-running, unlocked clocks)

v9_11 @8192, 113 kernels, 10 kHz sampling during kernel execution:
- GPC clock median 1.51 GHz (min 0.34 idle, max 1.98 boost spikes) — sustained load runs at
  ~76% of boost clock; this alone explains "75% of nominal peak".
- Tensor Active median 92% / mean 88.2% (matches ncu 89.1% at locked 1.43 GHz — bottleneck
  verdict is clock-independent).
- DRAM read 30% / write 5% (matches ncu DRAM 32.7%).
- Warps in flight 18% (matches ncu occupancy 18.2%).

Cross-validation: ncu (locked) and nsys (free) agree on every ratio within ~3% — ncu's
bottleneck attribution is trustworthy; only absolute TFLOPS differs (~3.6% slower under ncu).

## The complete "why are we at 75% of nominal peak" chain

1. Nominal 989.4T assumes 1.98 GHz boost; sustained H100 GEMM runs at ~1.51 GHz (power/thermal
   envelope) -> real ceiling ~749T for dense FP16 with F32 accumulate at this duty cycle.
2. Our top tier at 746T (kernel-only) = ~99.6% of that sustained ceiling.
3. cuBLAS at 739.7-779.1 = same physics, same envelope.
4. Tensor pipe 89% (ncu) / 92% (nsys): the remaining ~8-11% is mbarrier pacing between wgmma
   groups — the known cost of the 4-stage TMA pipeline design, not a fixable "inefficiency"
   within this design point.
