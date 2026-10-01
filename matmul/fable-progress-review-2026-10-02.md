# Fable progress review (2026-10-02, desk review of full post-fix table)

Prompt: ~/tmp/fable_progress_prompt.txt (full context + RESULTS-POSTFIX table + 6 questions).
Fable ran nothing (desk review, ~37 min think time). Verdicts below are condensed from the
full Chinese reply; verbatim reply preserved at the bottom.

## Verdicts

1. **Ladder healthy & unusually rigorous.** One mechanism per rung; null results kept and
   explained (v3/v5/v7/v9.10/v9.12/13 are the teaching gold). Measurement hygiene (tax-fix
   dual columns, nsys cross-validation at 94-101%, same-run cuBLAS, declared ~5% spread) is
   what most tutorial projects die on — this one didn't.
   Weak spots: v9_1–v9_4 missing from the ladder narrative (v9_1's 377T is the largest
   unexplained discontinuity, -25% vs v9 — explain or rule out build accident); only two
   sizes (4096/8192) — add 2048/3072/6144 to separate wave quantization from fixed overhead;
   v8/v8_1 warp-spec is a perf REGRESSION (187T < v6 208T) whose ladder role ("paves the road
   to v9's producer/consumer structure") must be stated explicitly.

2. **"Within noise of cuBLAS" is the right stopping point for this phase.** 989.4T is a
   boost-clock paper number; sustained load runs the clock at ≈75-80% of paper — cuBLAS
   itself at 73-75% of nominal confirms the physics. Real remaining headroom = CUTLASS best
   743T vs our 713T ≈ +4%. Meaningful next rungs: (i) 128x128 + cluster-2 multicast redo —
   the only mechanism learned-but-never-seen-winning in its correct geometry; (ii) persistent
   kernel / Stream-K — the last unexplored Hopper GEMM concept, targets the tail problem.

3. **Multicast-flat explanation refined**: not "geometry can't fit multicast" but "at 128x256
   the kernel is no longer L2-read-bound, so saving L2 reads buys nothing; 128x128 has ~2x
   L2 traffic per FLOP — that is the disease multicast treats". Falsifiable in 20 min with
   ncu (l2 sectors/flop, both geometries) once the dev-pod lands. A 128x256 A-multicast
   (cluster 1x2 along N) is theoretically possible but saves only half the traffic with the
   same implementation tax — expected flat/negative; we did NOT miss a viable design. Caveat:
   the CUTLASS 743T winner's edge may be partly more-CTAs-better-hiding, not pure multicast —
   the redo needs a same-geometry cluster-1 vs cluster-2 control.

4. **4096-vs-8192 gap = tail effects**: 512 CTAs = 3.88 waves at 4096 (last wave 88% full,
   ≤3% quantization cost) vs 2048 CTAs = 15.5 waves at 8192 (amortized); NOT TMA issue
   latency (would hurt both sizes proportionally). cuBLAS's better relative showing at 4096
   is likely a heuristic tile switch. BUT v9_8's pattern is reversed and single-run per-size
   percentages carry ±3-5% error bars (cuBLAS spread) — take 3-run medians of top tier
   @4096 before theorizing further.

5. **Recommended order: (b) ncu deep-dive → (c) 128x128 multicast redo → (a) tutorial
   capstone.** ncu is a force multiplier, not a peer option: without counters the tutorial
   writes conjectures, with them it writes evidence. Fill the v9_1–v9_4 narrative gap before
   the tutorial. Optional extras: Stream-K/persistent rung; nsys the cuBLAS kernel name to
   reverse-engineer its per-size config choice (30 min, directly explains Q4).

6. **Re-measure / rephrase list**:
   - v9_9 "100% of cuBLAS @8192" → must be phrased "statistical tie, 96-100%" (cuBLAS spread
     739.7/744.9/779.1); no single cell should be quotable without the spread note.
   - v9_1 377T: explain or rule out build accident.
   - v9_6 @8192 92% vs 82% @4096: largest size-sensitivity in the table; one independent
     re-run to confirm it's not single-run noise.
   - N=4096 L2-residency footnote: A+B = 64MB vs 50MB L2 — 100-iter loops may keep part of
     B resident, inflating ALL kernels incl. cuBLAS equally (harmless for relative compares,
     caveat for %peak @4096 absolute readings). 8192 (128MB each) unaffected. ncu L2 hit
     rate settles it.
   - wmma 28.1T < SIMT warptile 39T: normal for first-contact, but the tutorial must
     proactively explain (no pipeline, no swizzle → tensor core loses to tuned SIMT), else
     it's the reader's first "these numbers must be wrong".

---

## Full reply (verbatim, Chinese)

## Desk Review: GEMM Ladder (naive → WGMMA+TMA, sm_90a)

**1. 总体判断 — 这是一个健康且少见地严谨的 learning ladder。**

强项:
- 每一级只动一个 mechanism,且 null result(v3、v5、v7、v9.10、v9.12/13)全部保留并解释,而不是删掉——这是教学价值最高的部分。"cp.async 本身不买账、depth 才买账"(v5 vs v5.1)和 "更大 tile 不是免费的"(v7、v9.10)是教科书里找不到的第一手结论。
- 测量卫生到位:tax-fix 前后双列、nsys kernel-only 交叉验证(94-101%)、同 run 内对比 cuBLAS、明确声明 cuBLAS 的 5% spread。大多数 kernel 教程项目死在测量上,这个没有。
- CUTLASS calibration 作为外部锚点(cluster-1x1 best 481T vs 你们 692-713T)把 "我们的 no-cluster design point 是不是足够好" 变成了可回答的问题。

弱项:
- v9_1 到 v9_4 在 ladder 叙述里缺位(表里有数据、prose 里没机制描述),v9_1 的 377T 大跌没有解释。写 tutorial 前必须补齐。
- 只有 square N ∈ {4096, 8192} 两个点。scaling 结论(见 Q4)目前建立在两点连线上,至少补 2048/3072/6144 才能区分 wave quantization 和 fixed overhead。
- warp specialization(v8/v8_1)作为机制被引入后在 perf 上是倒退(187T < v6 的 208T),ladder 里要明确讲清它的角色是"为 v9 的 producer/consumer 结构铺路",否则读者会困惑为什么保留一个更慢的 rung。

**2. 停在 "within noise of cuBLAS" 是否正确 — 对本阶段,是,但有一个明确的下一级。**

先校准天花板:989.4T 是 boost-clock 纸面数,H100 SXM 持续负载下 clock 会落到让 sustained peak ≈ 75-80% 纸面值;cuBLAS 自己只到 73-75% 正是这个物理事实的印证。所以 "70-75% of nominal" 不是你们的 inefficiency,大部分是 clock reality。真实的剩余 headroom 是 CUTLASS best 743T vs 你们 713T ≈ +4%。

有意义的下一级(如果继续 perf 线):**128x128 + cluster-2 multicast 重做(即你们的 c 选项)**——它是 ladder 里唯一"机制已学、但从未在正确几何下见到正收益"的技术,CUTLASS 证据说它值 +155T(相对 cluster-1 同几何)。其次是 **persistent kernel / Stream-K scheduling**,它针对 Q4 的 tail/quantization 问题,也是 Hopper GEMM 知识图谱里你们还没碰的最后一块主要拼图。除这两个外,继续 shave 当前 design point 的回报趋近于零。

**3. multicast flat 的解释 — 方向正确,但我建议把归因再精确一步。**

你们说的 "128x128 两个 CTA 共享一个 B read、128x256 捕捉不到" 是现象层面对的,但更根本的机制是:**128x256 tile 的 arithmetic intensity 已经高到 kernel 不再 L2-read-bound,所以省 L2 读流量买不到时间;128x128 tile 每 flop 的 L2 流量约为两倍,multicast 省掉的是真实瓶颈。** 换句话说不是 "几何装不下 multicast",而是 "该几何下 multicast 治的病不存在"。这个版本的解释可以被 ncu 直接证伪/证实(l2 read sectors per flop,两种 tile 对比)——dev-pod 到位后值得花 20 分钟拿这个数。

128x256 下有没有该 work 的 multicast 设计?理论上 cluster 1x2 multicast A(沿 N 的两个 CTA 共享 A)可行,但 A 是小 operand(每 k-step 128x16 vs B 的 16x256),可省流量只有 B-multicast 方案的一半,而 implementation tax 不变——预期仍然 flat 或负。我认为你们没有漏掉可行方案;正确动作就是换 128x128 几何,即 Q5 的 (c)。另注意:CUTLASS 的 743T 赢家是 128x128 + 6-stage + cluster,它的优势可能部分来自更多 CTA → 更好 latency hiding,不全是 multicast,重做时要能分离这两个因素(同几何 cluster-1 vs cluster-2 对照)。

**4. 4096 vs 8192 gap 的解读 — 主因是 tail effects,但先压一下噪声。**

机制面:128x256 tile 下 N=4096 是 32x16=512 CTAs ≈ 3.88 waves(132 SM、每 SM 1 CTA),末波 88% 满——wave quantization 成本上限 ~3%,正好覆盖 94%→100% 的 gap;N=8192 是 2048 CTAs ≈ 15.5 waves,quantization 和 pipeline fill/epilogue tail 都摊薄了。cuBLAS 在 4096 相对更好,很可能是 heuristic 换了更合身的 tile。所以 pattern 指向:**fixed/tail 成本(wave quantization + prologue fill + epilogue tail),而不是 TMA issue latency**(那会两个 size 等比例伤)。学习动作:persistent/Stream-K 正是治这个的。

但注意:v9_8 的 pattern 是反的(97%@4096、94%@8192),而 cuBLAS 自身 spread 5%,意味着单 run 的 per-size 百分比自带 ±3-5% error bar。在为 "94 vs 100" 建理论前,top tier @4096 至少要 3 个独立 run 的 median——否则这可能是在解释噪声。

**5. 下一步排序(目标 = 掌握 Hopper GEMM):**

1. **(b) ncu deep-dive**——它不是和其他选项并列的,它是放大器:验证 Q3 的 L2-traffic 假说、Q4 的 tail 假说、v9_8 "unthrottle" 是否真的改变了 tensor pipe utilization。没有 counters,后面的 tutorial 写的是猜想;有了 counters,写的是证据。dev-pod 已在路上,顺序成本为零。
2. **(c) 128x128 multicast 重做**——ladder 唯一未完成的机制,且带着 ncu 证据去做(先测两几何的 L2 流量,预测收益,再验证),这一天的学习密度会是全项目最高的。
3. **(a) tutorial walk-through**——作为 capstone 最后做,此时每个 rung 的解释都有 counter 级证据背书。写之前先补 v9_1-v9_4 的机制叙述缺口。
4. (d) 可选延伸:Stream-K/persistent rung(补 scheduling 知识),或 nsys 抓 cuBLAS kernel name 反查它在两个 size 用的 config——半小时,能直接解释 Q4 的 cuBLAS 行为。

**6. 需要复测/重新表述的点(具体):**

- **v9_9 "100% of cuBLAS @8192"**:cuBLAS 自身 739.7/744.9/779.1 的 spread 下,这是 tie,不是 match。如果 cuBLAS 那天跑出 779,同一行就变成 95%。表述应统一为 "statistical tie, 96-100%"(你们在 prose 里已经这么说了,确保 RESULTS.md 的任何单格数字不被单独引用)。
- **v9_1 的 377T**:比 v9(505T)低 25%,是表里最大的未解释 discontinuity。要么补机制解释,要么确认不是 build/config 事故(参考你们自己的 "orphaned .cu / 失败 assert 留下半成品" 教训)。
- **v9_6 @8192 的 92%**(vs @4096 的 82%):全表最大的 size-sensitivity,方向上说得通(4-stage pipeline 在大 N 摊薄 fill cost),但 10pp 的跳变值得一次独立 re-run 确认不是单次波动。
- **N=4096 的 L2 residency**:A、B 各 32MB,L2 50MB——100 iters 连跑时 B(或 A)可能部分驻留 L2,均匀抬高所有 kernel 含 cuBLAS。对 kernel-vs-cuBLAS 的相对比较无害,但 "%peak @4096" 的绝对值解读要带这个脚注;8192(各 128MB)无此问题,这也可能解释部分 4096/8192 行为差异——ncu 的 L2 hit rate 一测便知。
- wmma 28.1T 低于 SIMT warptile 的 39T:对 first-contact kernel 正常,但 tutorial 里要主动解释(无 pipeline、无 swizzle 的 tensor core 跑不过精调 SIMT),否则是读者的第一个 "这数字错了吧"。
