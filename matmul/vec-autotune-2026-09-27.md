# Vectorized Autotune (FP32 + FP16) — Rung 6a

**Date**: 2026-09-27
**Kernel**: `matmul/matmul_vectorized_tuned.cu` (methods `vec_autotune_f32/f16`)
**GPU**: H100 80GB HBM3 (gcp5), node gcp5-h100-0-0, job 219293
**Config space**: 16 candidates mirroring archive `CANDIDATES_VEC`

## Results

| | winner config | TFLOPS |
|---|---|---:|
| FP32 | (128,128,16,8,8) | 33.49 |
| **FP16** | **(128,64,8,16,8)** | **36.53** |
| **FP16/FP32 (winners)** | | **1.091x** |

- Config 0 (default 128,128,8,8,8): FP32 32.74 / FP16 32.78 — both match the
  hard-coded baselines (32.73/32.83T) to 0.2%: template is faithful.
- FP16 tuning gain: +11.4% over hard-coded vectorized_f16 (32.83T).
- FP32 tuning gain: +2.3% over hard-coded vectorized (32.73T) — FP32 vectorized
  is already near its config-space optimum at the default.

## Findings

1. **FP16 decisively beats FP32 at the vectorized tuned level (1.091x)** — the
   first dtype win that survives tuning on both sides. Mechanism: the vectorized
   rung removed the load bottleneck (8-half loads), and tuning found the config
   where FP16's remaining advantages (halved SMEM footprint → more resident
   blocks) compound: winner (128,64,8,16,8) uses only 64 threads/block, so
   occupancy is limited by blocks/SM, where FP16's 3KB SMEM tile (vs 6KB FP32)
   doubles resident blocks and hides latency.
2. **FP32 collapses at 128-accumulator configs** (TM*TN=128: configs 1,2,4,5 all
   18-24T) while FP16 holds 30-35T on the same configs — with identical FP32
   accumulator counts. The asymmetry suggests FP16's smaller SMEM tiles change
   L1/register allocation enough to dodge the spill cliff; full SASS
   attribution deferred (would need ptxas -v spill counts per config).
3. Archive note: the ported `vectorized_auto` (34.65T, hard-coded archive build)
   vs our template FP32 best 33.49T — within 3.4%, consistent with template tax.
4. LAUNCH_FAIL(14) = 256x256 block, regs/SM exceeded (known class). The
   pre-fix run additionally had an OOB bug (stride-loop load mapping missed
   row<BM / row<BK guards when NUM_THREADS > tile loads/vec) — all-FP16 configs
   crashed; fixed same day, see git history.

## Data

CSV: `matmul/vec-autotune-f32-f16-gcp5-h100-2026-09-27.csv`

## Addendum: ABAB hard-coded confirmation (job 219295, node h100-0-9)

A = compile-time-constant launch of the winner (exported `launchVecWinnerF16`,
no dispatch chain), B = runtime dispatch path, same config (128,64,8,16,8).

- **B/A = 1.0000** (7 reps, batched == single, clock 1980, IQR < 0.03%) — the
  vec template has ZERO dispatch/codegen cost. The 1D rung's 7.5% template tax
  was a 1D-specific codegen artifact, not a general property of the
  template+dispatch design.
- **Official production number: 36.47T** (hard-coded build, ABAB protocol).
  Sweep's 36.53T reproduces to 0.2% — the vec sweep numbers are directly
  citable.
