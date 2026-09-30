# Fable 5 review of the kernel ladder (2026-09-30, claude-fable-5 via headless Hermes)

Prompt: ~/tmp/fable_kernel_review_prompt.txt (full ladder + protocol + beliefs).
Verbatim archive of the response follows.

---

## Verdict summary (Fable's own words, condensed)
- 515T / 52% MFU credible; protocol gaps: clock not locked (±3-5% soft edge),
  data entropy and cuBLAS baseline call to double-check.
- Gap attribution DISAGREES with us: setmaxnreg register rebalance (#1) >
  persistent ping-pong + async epilogue (#2, +50~80T) > cluster+multicast
  (#3, +25~50T). Our multicast bet ranks third.
- v9_8 negative result OVERTURNED: we falsified "naive persistent with
  per-tile full-CTA drain", not persistence itself. Must revisit as ping-pong.
- v9_4 labeled INCOMPLETE not negative (bijection bug unfixed, 428.6 not a
  clean datapoint).
- v9's +140T has an unseparated confound: issue-slot relief AND swizzled
  operand reads (bank-conflict elimination) in one diff.
- Missing pedagogical rungs: mma.sync m16n8k16 + ldmatrix (between WMMA and
  WGMMA), setmaxnreg, persistent ping-pong, cluster/multicast, split-K.
- Endgame projection: 700±30T (~92% cuBLAS) with #1+#2+#3; the last 60T is
  cutlass instruction-scheduling polish, not worth chasing in a learning repo.

## Full response (verbatim)

senior kernel engineer 帽子戴上,直接说。整体判断先给:这是一条罕见地干净的 learning ladder,515T/52% MFU 是可信且相当不错的数字,但你们的 gap 归因排序是错的——最大的一块不是 TMA multicast,而是 register budget + persistent ping-pong scheduling。下面逐题。

1) Sanity check:数字可信吗?
可信。公开的 H100 手写 HGEMM worklog(Pranjal Shankhdhar 的系列、cutlass 复现类项目)在"wgmma + TMA + multi-stage、无 cluster、无 persistent scheduling"这个设计点上普遍落在 450–550T,你们 515T 恰好在该区间上沿。cuBLAS 762T = 77% MFU 也是 H100 8192 方阵的典型值。没有哪个数字让我皱眉。
测量协议上三个真正的 red flag:
a) Clock 没锁。你们报 1.98GHz 但 H100 SXM 在 700W 限下跑 dense FP16,cuBLAS(77% MFU)比你们的 kernel(52% MFU)更容易触 power/thermal throttle。100 runs 连跑,先测谁、GPU 冷热,能差 3-5%。用 nvidia-smi -lgc 锁 clock,或至少记录每次 run 的实际 SM clock,否则"67.6% of cuBLAS"这个比值本身有 ±4% 的软边。
b) 数据熵。GEMM 的功耗(→throttle)强依赖 operand 的 bit 熵。如果你们的 kernel 和 cuBLAS 用同一份 random data,OK;如果任何一边用过 zeros/ones init,数字会虚高。确认一下。
c) cuBLAS baseline 的 accumulate 类型。你们 wgmma 用 f32 accumulator、C 写 FP32——公平的 baseline 是 cublasGemmEx(F16 in, F32 compute, F32 out),不是 cublasHgemm(F16 accumulate)。H100 上两者峰值都是 989T 所以差别不大,但 epilogue 带宽不同(C 是 2B vs 4B),8192 时 C 写流量差一倍。确认你们调的是哪个。
另外 belief #1 有一个 confound 你们没拆:v9 的 +140T 不只是"768 cp.async → 6 TMA 的 issue-slot 解放",SWIZZLE_128B 同时消灭了 wgmma 从 SMEM 读 operand 的 bank conflict(pre-v9 你们大概率是 padding 或裸布局)。两个机制混在同一个 diff 里。不值得回头补实验,但 post-mortem 文档里应该写成"issue pressure + swizzled operand reads,未分离"。

2) Trajectory 评价
顺序基本教科书级正确:SIMT ladder → roofline 撞墙 → TC → 喂料问题 → TMA → rasterization → epilogue → pipeline depth。特别赞:v3 用 AI=32×3.35TB/s≈107T 算出 64x64 tile 的 roof 然后立刻换 shape,v9_3 用 N-sweep 证明 L2 假设——这是多数工业 worklog 都懒得做的。
浪费的功:v8/v8.1(cp.async warp-spec)。在 1-2 CTA/SM、consumer spin-wait 的设计点上这是可预判的死路——cp.async 时代的 warp-spec 本来就只在高 occupancy 下成立。但作为 learning project 它教了 mbarrier 语义,为 v9 铺路,所以我算它"学费"而非"浪费"。
真正的顺序错误只有一个:v9_8(persistent)做在 epilogue overlap 之前。persistent 的全部意义就是让 tile i 的 epilogue 和 tile i+1 的 math 重叠;你们先做了 naive per-tile drain 的 persistent,测出负结果,然后归因给"persistence 不值"。这是把实现缺陷当成了机制结论(见 Q4)。
缺的 rung:见 Q5。

3) 250T gap 的机制排序(这是我最不同意你们的地方)
你们押 TMA multicast 是大头。我不同意。先算账:8192 时你们 515T,GROUP_M=8 下的 DRAM 流量离 3.35TB/s roof 很远——你们不是 DRAM-bound,multicast 省的是 L2/fabric 带宽,是二阶收益。真正的排序:
#1 Register budget / setmaxnreg —— 立刻做,最便宜。
算术:64K regs/SM ÷ 384 threads = 170 regs/thread。你们每个 consumer thread 光 m64n256 f32 accumulator 就占 128 regs,剩 ~42 个给地址、descriptor、循环变量——编译器几乎必然在 spill 或者在 wgmma 间隙插 stall。而 producer wg 那 128 个线程几乎不用寄存器,却平摊走了 21K regs。
实验(v9_9):setmaxnreg.dec producer → 40,setmaxnreg.inc consumers → 232。这正是 cutlass sm90 kernel 的标准动作。预期 +30~60T。一行 PTX 级改动,先做这个。
#2 Persistent + ping-pong consumer warpgroups —— gap 里最大的单块。
cuBLAS 在这个 size 上跑的就是 cutlass 风格 persistent ping-pong:两个 consumer wg 交替,一个做 tile i 的 epilogue 时另一个已在算 tile i+1 的 wgmma,epilogue 完全藏进 math 影子里;TMA store 走 async bulk group,不需要你们那两次 __syncthreads 全 CTA drain。
实验(v9_10):v9_8 的骨架 + 每个 consumer wg 独立 tile 循环 + mbarrier 换掉 __syncthreads + epilogue 用 cp.async.bulk 不等 wait 0 才进下一 lap。预期 +50~80T @8192。
#3 Cluster(2 CTA)+ TMA multicast —— 有效但排第三。
省一半 B 的 L2 读流量,主要缓解 fabric 姿态、让 L2 给 A 腾容量。预期 +25~50T,且与 #2 正交可叠加。cluster launch + mbarrier cluster scope 的工程量比 #1 大一个数量级,所以顺序放后。
#4 Epilogue stmatrix 化 —— 小。你们 v9_5 已经把 scattered 8B writes 干掉了,SMEM staging round-trip 剩余成本在 non-persistent 下只发生一次/CTA。在 #2 落地后这块基本免费,单独做预期 <10T。
三件叠满,700±30T 是现实的落点,即 ~92% cuBLAS。最后 60T 是 cutlass 多年的 instruction scheduling 微调,学习项目不必追。

4) Negative results 复核
- v8 warp-spec cp.async:post-mortem 正确,同意归档。
- v9_1 128x128:AI 84→64 的归因正确,且你们自己抓到了 confound,好。
- v9_4 2D rect swizzle:结论方向对(收益本来就小),但严格说这是"未完成"不是"negative"——bijection bug 没修,428.6 不是干净数据点。归档时标签要改成 INCOMPLETE,别让未来读代码的你把它当成已证伪。
- v9_8 persistent:不同意你们的结论。你们证伪的是"naive persistent + 全 CTA drain epilogue",不是 persistence 本身。"launch overhead amortized over 64 k-laps"这句话是对的但打错了靶子——persistent 的收益从来不是省 launch,是 epilogue/prologue 与 math 的跨 tile 重叠(以及 tile 调度自由度)。这条必须 revisit,见 #2。

5) Pedagogy
v2→v9.7 作为教学序列是好的,叙事线清晰:每一步都有"撞什么墙→换什么机制"。要补的 rung:
a) WMMA 和 WGMMA 之间缺 mma.sync m16n8k16 + ldmatrix 一档。学生直接从 WMMA API 跳到 wgmma descriptor,中间"寄存器 fragment 布局到底长什么样"这个概念没有落地点。这一档也自然引出 stmatrix。
b) setmaxnreg(做了 #1 就有了)——教"寄存器是 warp-spec 设计的一等资源"。
c) persistent ping-pong(#2)——教"scheduling 也是 kernel 的一部分"。
d) cluster + multicast(#3)——教 sm90 的 CTA 间 CTA 间协作。
e) 尾声可选:split-K/stream-K,一句话讲清"方阵之外的世界"(skinny GEMM 时你们整条 ladder 的结论都会翻)。

6) Wildcard 掯名(>30T 候选)
1. setmaxnreg rebalance:+30~60T,工作量半天。
2. persistent ping-pong + async epilogue:+50~80T @8192,工作量一到两周。
3. cluster 2x1 + TMA multicast on B:+25~50T,工作量一周。
4. 锁 clock 重测全表:不涨性能,但可能"白捡"或"白丢"3-5% 的报告数字——诚实起见先做。
5. cudaAccessPolicyWindow / L2 residency、cp.async.bulk.prefetch:各 <10T,不值得当 rung,可当脚注。
没有其他 H100 秘技了——sm90 的牌就这几张:TMA(打了)、wgmma(打了)、swizzle(打了)、setmaxnreg、cluster/multicast、persistent scheduling。打完这三张你们就是 cutlass 的手写复刻,ladder 圆满。
