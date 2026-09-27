# Autotune Classes Ported from archive-cudakernels

**Date**: 2026-09-26
**Source**: `~/code/archive-cudakernels` main (commits 4c01b74 et al.) — the
2026-05/06 autotune work that never migrated when this repo was created
(only docs were migrated in 95f1a7d).

## What was ported

| Method | Class | Verified (gcp5, job 219254, node h100-0-19) | Old-repo direct run (job 219227) |
|---|---|---:|---:|
| `1d_blocktile_auto` | Matmul1DBlocktileAuto | 19.09T | 19.22T |
| `2d_blocktile_auto` | Matmul2DBlocktileAuto | 33.80T | 34.07T |
| `vectorized_auto` | MatmulVectorizedAuto | 34.65T | 34.93T |
| `warptile_auto` | MatmulWarptileAuto | **39.07T** | 39.30T |

All within ±1% of the archive binary on the same cluster — the port is
faithful. Files: `matmul/matmul_{1d_blocktile,2d_blocktile,vectorized,warptile}_auto.{cu,h}`
(baseline classes stripped, Auto classes + templated kernels intact).

## Headline: warptile_auto 39.07T > warptile_dbuf 37.59T (pending same-session)

The archive warptile autotune reaches 39T — ABOVE our dbuf winner's 37.6T —
on a different node/session. A same-session comparison is queued (job
219256). If it holds, the tuned warp-tiling optimum beats double buffering
on H100 FP32, which would reorder the ladder's top rung.

## Porting pitfalls hit (all fixed, all committed)

1. Baseline-stripper script ate the Auto constructors' signature lines
   (orphan member-init lists) — restored.
2. Python-replace registration missed silently (compound strcmp anchor) —
   methods were "Unknown method" until patched by hand.
3. Baseline `__global__` kernels duplicated across TUs (multiple definition
   at link) — removed from auto TUs.

Lesson recorded: scripted code transforms need compile+link verification in
the same step, not after the fact.
