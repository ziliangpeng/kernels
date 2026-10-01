# RESULTS — kernel-only (nsys cuda_gpu_kern_sum, 2026-10-01)

Discovery: the bench harness taxes every FP16-in kernel family with
per-iteration convert+convert+transpose (~188us at 4096; ~750-770us at
8192). cuBLAS FP16 pays it ONCE (constructor). All pre-2026-10-01
RESULTS.md rows for FP16-in families are END-TO-END and tax-contaminated
(kernel-only is the industry-standard headline; keep both).

Sweep: nsys profile per variant, N=4096/8192, main kernel = largest
non-tax compute kernel; TFLOPS from MEDIAN duration, 110 instances.
Node gcp5-h100-0-9. Tax mode verified by instance counts
(percall = convert kernels x220/110 iters; once = x2).

## N=4096 (FLOPs = 137.44G)

| rank | variant | kern-med | kernel-only TFLOPS | % of 989.4T peak | tax mode |
|---|---|---|---|---|---|
| 1 | cuBLAS FP16 | 191.6us | 717.1 | 72.5% | once |
| 2 | wgmma_v9_11 | 194.1us | 708.0 | 71.5% | percall |
| 3 | wgmma_v9_8 | 197.3us | 696.7 | 70.4% | percall |
| 4 | wgmma_v9_9 | 197.3us | 696.4 | 70.4% | percall |
| 5 | wgmma_v9_7 | 198.5us | 692.3 | 70.0% | percall |
| 6 | wgmma_v9_13 | 204.1us | 673.4 | 68.1% | percall |
| 7 | wgmma_v9_6 | 231.3us | 594.2 | 60.1% | percall |
| 8 | wgmma_v9_5 | 237.1us | 579.8 | 58.6% | percall |
| 9 | wgmma_v9_12 | 250.1us | 549.5 | 55.6% | percall |
 | | wgmma_v9_10 | 252.7us | 543.9 | 55.0% | percall |
 | | wgmma_v9_4 | 261.4us | 525.7 | 53.1% | percall |
 | | wgmma_v9_2 | 262.8us | 523.0 | 52.9% | percall |
 | | wgmma_v9_3 | 267.4us | 514.0 | 51.9% | percall |
 | | wgmma_v9 | 272.4us | 504.6 | 51.0% | percall |
 | | wgmma_v9_1 | 367.7us | 373.8 | 37.8% | percall |
 | | wgmma_v6 | 653.7us | 210.3 | 21.3% | percall |
 | | wgmma_v7 | 665.6us | 206.5 | 20.9% | percall |
 | | wgmma_v8 | 733.6us | 187.4 | 18.9% | percall |
 | | wgmma_v8_1 | 768.7us | 178.8 | 18.1% | percall |
 | | wgmma_v4 | 832.1us | 165.2 | 16.7% | per2call |
 | | wgmma_v5 | 871.3us | 157.7 | 15.9% | percall |
 | | wgmma_v2 | 1143.7us | 120.2 | 12.1% | percall |
 | | wgmma_v3 | 1186.3us | 115.9 | 11.7% | percall |
 | | wmma | 4838.7us | 28.4 | 2.9% (of 989.4) | percall |
 | | cublas (FP32) | 2625.9us | 52.3 | 78.1% of 67T | none |
 | | warptile_dbuf | 3628.2us | 37.9 | 56.6% of 67T | none |
 | | vectorized | 4171.4us | 32.9 | 49.1% of 67T | none |
 | | warptile | 4841.9us | 28.4 | 42.4% of 67T | none |
 | | 2d_blocktile_f16 | 6357.4us | 21.6 | 2.2% (of 989.4) | once |

## N=8192 (FLOPs = 1099.5G)

| variant | kern-med | kernel-only TFLOPS | % peak | vs cuBLAS FP16 |
|---|---|---|---|---|
| cuBLAS FP16 | 1411.3us | 779.1 | 78.7% | 100% |
| wgmma_v9_11 | 1473.5us | 746.2 | 75.4% | 95.8% |
| wgmma_v9_7 | 1475.0us | 745.5 | 75.3% | 95.6% |
| wgmma_v9_9 | 1475.4us | 745.3 | 75.3% | 95.6% |
| wgmma_v9_8 | 1489.2us | 738.3 | 74.6% | 94.7% |
| wgmma_v9_13 | 1503.5us | 731.3 | 73.9% | 93.9% |
| wgmma_v9_6 | 1584.2us | 694.0 | 70.1% | 89.1% |
| wgmma_v9_5 | 1733.4us | 634.3 | 64.1% | 81.4% |
| wgmma_v9_3 | 1786.2us | 615.6 | 62.2% | 79.0% |
| wgmma_v9_12 | 1791.8us | 613.6 | 62.0% | 78.8% |
| wgmma_v9_10 | 1865.0us | 589.6 | 59.6% | 75.7% |

## Narrative rewrite

- v9.7 kernel-only = 692.3T @4096 / 745.5T @8192 = 70.0% / 75.3% of
  nominal FP16 peak; **95.6% of cuBLAS @8192** (was "68% of cuBLAS").
- New champion: v9_11 (708.0T @4096, 746.2T @8192) — unthrottle fix wins
  on clean accounting (was flat under tax).
- The old "+87% v9 over v6" was tax-compressed; kernel-only v9/v6 =
  504.6/210.3 = +140%.
- Old per-lap anomaly (1.5us@4096 vs 1.07us@8192) was tax illusion
  (Fable model: multiplier ratio 1.33 vs observed 1.40; kernel-only
  per-lap: 198.5us/64laps = 3.10us vs 1475us/128 = 11.5... per-lap here
  is per-CTA-lap across tiles; cross-N comparison now clean at ~same
  utilization: 70% vs 75%).
- cuBLAS FP16 pays convert ONCE (instance count = 2, not 220): all
  historical v9-vs-cuBLAS gaps overstated the tax on our side only.
- wmma percall-tax = 113.5us/iter (convert-only, no transpose: it reads
  B row-major) — even the FP32-in WMMA path was taxed.
- FP32 families (naive..warptile, cublas FP32): zero tax — their old
  numbers stand unchanged.

## CUTLASS unfiltered re-run @4096 (Fable-prompted; job 221805, 4204 blocks)

| reference | TFLOPS @4096 | note |
|---|---|---|
| CUTLASS best overall | **743.2** | cta 128x128 clu 2x1 st 6 |
| CUTLASS cluster 1x1 best | 481.4 | (256x128 tile) |
| our v9_11 (kernel-only) | 708.0 | 95.1% of CUTLASS best |
| our v9.7 (kernel-only) | 692.3 | 93.1% of CUTLASS best |
| cuBLAS FP16 | 717.1 | 96.5% of CUTLASS best |

Confirms Fable's model: the earlier 512.6T "1x1 ceiling" was a single-row
artifact (real 1x1 best at 8192 was also ~481-512 range, and the overall
best needs cluster multicast on the 128x128 tile). Note this build still
has NO 128x256 f16 kernels; the closest geometry to ours (256x128) tops
at 481.4 in cluster 1x1 — our 128x256 tile at 692.3T (cluster 1x1 by
construction) EXCEEDS every same-cluster CUTLASS config in this build.
v9_11 = 95.1% of the absolute CUTLASS best (which uses multicast+2-CTA
clusters we showed cannot pay on 128x256).

## Final standings (kernel-only, @4096 / @8192)

  v9_11   708.0 / 746.2   <- new champion
  v9.7    692.3 / 745.5
  cuBLAS  717.1 / 779.1
  CUTLASS 743.2 / 667.8*  (*8192 number from the earlier f16-subset sweep)

Our hand-written kernel sits at 93-96% of cuBLAS and ~95% of CUTLASS
best, at 70-75% of nominal peak, on a design point (128x256, no cluster)
where CUTLASS's own best same-constraint config reaches only 481T.
