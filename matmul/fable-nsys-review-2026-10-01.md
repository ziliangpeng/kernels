# Fable 5: nsys profiling review (2026-10-01)

Prompt: ~/tmp/fable_nsys_prompt.txt. Context: nsys probe on v9.7 @4096
revealed the bench harness taxes every v9-family iteration with
convert+convert+transpose (56.3+56.3+75.1us) around a 198.1us main kernel.

## Verdicts (full text in session log; key points)

1. MATH: 693.5T kernel-only is SELF-CONSISTENT and physically feasible.
   Quantized ceiling: 512 CTAs/132 SMs = 3.879 waves -> 4 waves x 35.8us
   ideal per-CTA = 143.3us -> ceiling 959T (97% of 989.4T peak). We are at
   72.3% of the quantized ceiling, 70.1% of nominal peak — the normal
   band for a hand-written no-multicast kernel (CUTLASS same-tier 72-80%).
   The 512.6T "CUTLASS 1x1 ceiling" is a PSEUDO-ceiling (single config row,
   51.8% of peak — likely layout/subset artifact); re-run cutlass_profiler
   unfiltered at 4096, expect best 700-800T.
2. HEADLINE METRIC: kernel-only TFLOPS is the industry standard (CUTLASS
   profiler, all GEMM papers). Keep both columns (end-to-end / kernel-only
   + prep-tax us) with a footnote: transposeB is a layout CONSTRAINT of
   variants that need it, note it, don't re-rank for it.
3. NARRATIVE REWRITE: v9.7 kernel-only 693.5T = ~96.8% of cuBLAS 716.6T
   (same-harness cublasGemmEx likely has no per-call tax — VERIFY via nsys
   kern_sum: instance counts answer it, no guessing). The "+87% v9 vs v6"
   jump is partly tax illusion — kernel-only re-ranking will INCREASE
   gaps (tax penalized faster kernels harder). Old table: keep, label
   "pre-2026-10-01: end-to-end, tax-contaminated".
4. 4096-vs-8192 per-lap anomaly (1.5 vs 1.07us) is mostly TAX ILLUSION:
   tax multiplier 1.95x @4096 vs 1.47x @8192, ratio 1.33 vs observed 1.40
   — explained; kernel-only per-lap expected flat (residual ~5% wave+L2).
5. NEW WEAPON: nsys --gpu-metrics-device=0 (10kHz sampling) gives
   Tensor Pipe Active % (poor-man's sm__pipe_tensor), DRAM bw, and
   MEASURED SM clock — H100 SXM sustains 1.6-1.8GHz under MMA load
   (700W), so nominal 989.4T peak is unreachable in practice; with
   measured clock our 70.1% becomes 75-85% pipe utilization. MUST column.
6. SURVEY METHODOLOGY: nsys does not perturb kernel durations (GPU
   hardware timestamps; overhead lands on CPU launch path). The 2%
   bench<sum anomaly: kern_sum averages all 110 (incl. warmup ramp);
   report median/min or filter by NVTX range. v2-v7 tax check via
   instance counts (220 converts = per-call; 2 = constructor-only).
   FP32 families' peak is 67T not 989.4T — add "% of respective peak".
7. GO-FORWARD RANKING: (1) harness fix (convert/transpose to
   constructor) + full nsys sweep WITH gpu-metrics — one batch; (2) K-sweep
   scaling (fixed M,N, K 512..16384; slope = per-lap mainloop, intercept
   = prologue+epilogue) — sharpest zero-permission mainloop microscope;
   (3) clock64 in-kernel phase timestamps (intra-CTA only); (4) NVTX for
   warmup/timed windowing; (5) admin email for ncu perms, dev-pod route
   after that.
8. HONESTY CHECK: after sweep, diff C against cuBLAS once — rule out any
   silent early-exit making 198.1us look too good.
