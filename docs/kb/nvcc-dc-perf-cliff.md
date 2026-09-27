# nvcc -dc (Relocatable Device Code) Cost 30-60% GEMM Perf

**Date**: 2026-09-26, gcp5 H100 (SM90)
**Found via**: the 1D-blocktile autotune discrepancy hunt.

## Symptom

Same kernels, same node, same harness: new-repo binary measured 12.5/18.4/
20.5/26.9T (1d/2d/vectorized/warptile) while the archive-cudakernels binary
measured 17.5/22.2/32.7/29.4T — a 30-60% systematic gap on the FP32 ladder.

## Root cause

The parallel incremental build script compiled each .cu with `-dc`
(relocatable device code) + a final `nvcc -dlink`-style link, because it was
assumed cross-TU template launches needed it. They do not: every dispatch
table in this repo is `#include`d into the TU that launches it, so whole-
program device code is unnecessary. With `-dc`, nvcc compiles device code to
relocatable PTX/cubin that is linked later — which disables several
optimizations (notably aggressive inlining/optimization across the
device-link boundary and some ptxas passes), producing measurably slower
kernels.

## Fix

Compile with plain `-c` (no `-dc`), link objects normally. Verified: default
configs recover to archive-binary levels (see full-ladder-rebench follow-up).

## Rule

- Separate-TU device compilation is only needed for true cross-TU device
  calls (e.g., `__device__` functions in a library TU called from another TU).
- If dispatch is via #include'd tables into the launching TU, never use -dc.
- When a binary is mysteriously uniformly slower than an old one on the same
  GPU, diff the BUILD FLAGS FIRST — before suspecting kernels, harness, or
  nodes. (This cost us a day of "harness/median vs mean" theorizing; the
  archive binary rebuilt with the new flags would have isolated it in
  minutes.)
