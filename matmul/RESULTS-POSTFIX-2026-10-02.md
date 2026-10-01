# RESULTS — post-tax-fix full sweep (2026-10-02, job 221814, node gcp5-h100-0-9, fixed harness)

All 25 methods x N=4096/8192, same node, same binary, one job, 100 timed iterations each.

## Column definitions (READ THIS FIRST)

- **old** — end-to-end GFLOPS before the 2026-10-01 harness tax fix. The old harness re-ran the F32->F16
  convert (x2) + transpose kernels EVERY bench iteration (~188us/iter @4096, ~760us/iter @8192), while
  cuBLAS converted once at construction. "old" numbers are kept for history only — they systematically
  understate every FP16-in kernel vs cuBLAS by ~1.5x @8192.
- **NEW** — end-to-end GFLOPS with the fixed harness (one-time prep, guarded by input-pointer change).
  This is the current truth. Bench end-to-end now converges to nsys kernel-only medians (see nsys% column).
- **nsys%** — NEW / nsys kernel-only median (nsys `cuda_gpu_kern_sum`, job 221803, 110 instances).
  94-101% everywhere = the two independent measurement paths agree; the numbers are trustworthy.
- **%cuBLAS** — NEW / cuBLAS_fp16 in the SAME job (724.7T @4096, 739.7T @8192). Note cuBLAS itself has
  ~5% run-to-run spread (739.7 / 744.9 / 779.1 observed @8192 across jobs), so 94-100% is a statistical tie
  at the top.
- **%peak** — fraction of the H100 SXM FP16 dense-precision peak (989.4T = 1979T sparse / 2, 132 SMs
  @ 1.98 GHz boost). The 4096 column uses the same denominator.

## Full table (GFLOPS unless marked; "—" = not measured at that size)

| method | old @4096 | NEW @4096 | nsys% @4096 | old @8192 | NEW @8192 | nsys% @8192 | %cuBLAS @4096 | %cuBLAS @8192 | %peak @4096 | %peak @8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `wmma` | 27.6 | **28.1** | — | — | **29.3** | 100% | 4% | 4% | 2.8% | 3.0% |
| `wgmma_v2` | 104.3 | **120.3** | 100% | — | **122.3** | 100% | 17% | 17% | 12.2% | 12.4% |
| `wgmma_v3` | 100.3 | **115.4** | 100% | — | **120.5** | — | 16% | 16% | 11.7% | 12.2% |
| `wgmma_v4` | 135.1 | **164.8** | 100% | — | **173.4** | — | 23% | 23% | 16.7% | 17.5% |
| `wgmma_v5` | 130.6 | **157.2** | 100% | — | **164.3** | — | 22% | 22% | 15.9% | 16.6% |
| `wgmma_v5_1` | 142.5 | **175.5** | — | — | **187.0** | — | 24% | 25% | 17.7% | 18.9% |
| `wgmma_v6` | 163.2 | **208.3** | 99% | — | **225.8** | — | 29% | 31% | 21.1% | 22.8% |
| `wgmma_v7` | 163.4 | **207.5** | 100% | — | **221.1** | — | 29% | 30% | 21.0% | 22.4% |
| `wgmma_v8` | 150.5 | **187.0** | 100% | — | **200.4** | — | 26% | 27% | 18.9% | 20.3% |
| `wgmma_v8_1` | 145.8 | **178.9** | 100% | — | **192.0** | — | 25% | 26% | 18.1% | 19.4% |
| `wgmma_v9` | 305.0 | **505.7** | 100% | 426.8 | **542.4** | 100% | 70% | 73% | 51.1% | 54.8% |
| `wgmma_v9_1` | 253.2 | **377.3** | — | — | **388.6** | — | 52% | 53% | 38.1% | 39.3% |
| `wgmma_v9_2` | 311.4 | **522.4** | 100% | 426.8 | **561.3** | — | 72% | 76% | 52.8% | 56.7% |
| `wgmma_v9_3` | 309.5 | **514.8** | — | 452.4 | **590.1** | 96% | 71% | 80% | 52.0% | 59.6% |
| `wgmma_v9_4` | — | **526.6** | — | 428.6 | **559.0** | — | 73% | 76% | 53.2% | 56.5% |
| `wgmma_v9_5` | 331.6 | **581.5** | 100% | 460.9 | **604.1** | 95% | 80% | 82% | 58.8% | 61.1% |
| `wgmma_v9_6` | 336.7 | **597.2** | 101% | 491.9 | **681.9** | 98% | 82% | 92% | 60.4% | 68.9% |
| `wgmma_v9_7` | 367.1 | **696.2** | 101% | 514.8 | **713.9** | 96% | 96% | 97% | 70.4% | 72.2% |
| `wgmma_v9_8` | 367.2 | **704.5** | 101% | 508.7 | **697.1** | 94% | 97% | 94% | 71.2% | 70.5% |
| `wgmma_v9_9` | 368.9 | **700.8** | 101% | 511.4 | **742.2** | 100% | 97% | 100% | 70.8% | 75.0% |
| `wgmma_v9_10` | 318.6 | **543.1** | 100% | 432.4 | **569.3** | 97% | 75% | 77% | 54.9% | 57.5% |
| `wgmma_v9_11` | 370.8 | **713.1** | 101% | 510.6 | **711.9** | 95% | 98% | 96% | 72.1% | 71.9% |
| `wgmma_v9_12` | 320.0 | **548.7** | 100% | 443.1 | **580.3** | 95% | 76% | 78% | 55.5% | 58.7% |
| `wgmma_v9_13` | 359.8 | **681.0** | 101% | 504.6 | **688.4** | 94% | 94% | 93% | 68.8% | 69.6% |
| `cublas_fp16` | 716.6 | **724.7** | 101% | 762.0 | **739.7** | 95% | 100% | 100% | 73.2% | 74.8% |

## Reading notes

1. Top tier @8192: v9_9 742.2T (100% of same-run cuBLAS), v9_7 713.9T (97%), v9_11 711.9T (96%).
   cuBLAS run-to-run spread ~5% -> statistical tie at the top.
2. %peak: best 75.0% (v9_9 @8192); cuBLAS itself 74.8% — same line.
3. Ladder visible: wmma 29 -> wgmma 120 -> v6 226 -> v9 (TMA) 542 -> v9_9 742.
4. Old-vs-new gap ratio @8192 ~1.4-1.5x = the removed per-iteration tax (kernel 1475us + tax 760us).
5. nsys kernel-only reference table: RESULTS-KERNELONLY-2026-10-01.md. CUTLASS references:
   743.2T @4096 (128x128 clu 2x1 multicast, unfiltered sweep job 221805); 512.6T @8192 was a single-row
   artifact of the 1x1-filtered sweep — the honest cluster-1x1 best @8192 is 663.7T (cluster 1x2).

## Provenance

- job 221814 (this table), job 221808 (tax-fix convergence spot check), job 221803 (nsys sweep),
  job 221805 (CUTLASS unfiltered). All on gcp5 Slurm, GPU-exclusive H100 nodes.
- Harness fix commit: "perf(harness): fix per-iteration convert/transpose tax". Table commit: this file.
