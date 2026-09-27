# ABAB 定稿实验:typed vs tuned 同配置的真实差距 — 7.5% codegen 差,非测量 artifact

**Date**: 2026-09-27
**Tool**: `matmul/abab_bench.cu` (new), `scripts/build_abab.sh`
**GPU**: H100 80GB HBM3, gcp5, node gcp5-h100-0-19, job 219278
**Question (Ziliang's)**: 能否精确测量?同节点同 GPU 同进程,CUDA event 计时,warmup 充分,排除一切非 kernel 开销。tuned 慢于 not-tuned 到底是不是真的?

## Protocol

- 同进程交替 ABAB × 7 轮(A=typed 硬编码, B=tuned 模板 dispatch,同 config 64,64,8,8,同 FP16 storage + FP32 accum)
- CUDA event 两种计时法都测:batched(100 连发一对 event)和 single(每次 launch 一对 event,取 median)——Ziliang 的经典配方
- 每 warmup 10 次 + sync,计时区外;constructor/convert 计时区外
- verify A/B 输出逐位一致(maxdiff = 0.000e+00)
- NVML 每 rep 采样 SM clock:恒定 1980 MHz
- 归因探针:B 的 host launch 2.6 μs/launch vs GPU 8580 μs → GPU-bound,dispatch 链完全隐藏

## Results (N=4096, median of 7, IQR in ms)

| | batched | single | TFLOPS |
|---|---|---|---|
| A typed | 7.9772 (IQR 0.0016) | 7.9807 (IQR 0.0037) | 17.23 |
| B tuned | 8.5791 (IQR 0.0024) | 8.5827 (IQR 0.0027) | 16.02 |
| **B/A** | **1.0755** | **1.0754** | |

- batched ≈ single(差 0.05%)→ launch gap 不是因素;两种计时法互相印证
- IQR 0.02-0.03% → 测量精度极高,7.5% 差距是真信号
- **B(模板构建)在同 config 下比 A(硬编码构建)慢 7.55%,纯 codegen 差异**

## 结论(推翻之前两版归因)

1. **tuned < not-tuned 是真的,且根因是构建方式**:模板 dispatch 构建生成的 SASS 比硬编码构建慢 7.5%。之前归因的「2.4% harness 代价」实测是 7.5%;「节点差 4.8%」也存疑(今天 node 19 跑出 17.23T,与 node 15 的 17.21T 一致——早前 16.42T 那次疑为 clock/热状态,当时未采样 clock)
2. **autotuner 逻辑本身无错**:它在模板构建的候选里选了最优(16.31 > 16.03)。但整个候选空间都背着 7.5% 的模板税,所以永远输给硬编码构建的默认 config
3. **正确用法**(= Fable 规则 6 的实证):autotuner 输出 config → 用硬编码路径重建该 config 的 kernel → 在 baseline 路径复测。tuning 的意义是找 config,不是当生产 kernel
4. 测量方法学:Ziliang 的 CUDA-event 配方完全成立,single 与 batched 互相印证到 0.05%——「测不准」不成立,之前是比错了对象(跨 job/跨节点/跨 binary)

## Open follow-up

- SASS diff(matmul1DTunedKernel<Half,64,64,8,8> vs 硬编码 kernel):7.5% 从哪来(指令数?寄存器?bank conflict?)——修好 codegen 后模板税可望归零,autotune 才能真正兑现
- 之前「FP16 tuned 16.31 vs typed 17.23」的完整叙事已按本实验修正
