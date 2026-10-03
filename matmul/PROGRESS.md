# 学习进度：H100 matmul 优化阶梯

目标：从 naive kernel 一路学到 `wgmma_v9_x`（约 cuBLAS 的 96%），真正理解每一步为什么快。这是长期计划，不赶进度。

最后更新：2026-10-03

---

## 怎么用这个文件

每一课在本目录有三个文件：

| 文件 | 内容 |
|---|---|
| `<kernel>.cu` / `.h` | kernel 源码 |
| `<kernel>.html` | 可视化页面（交互图、算账、实测），浏览器打开 |
| `<kernel>.notes.md` | 学习笔记：完整讲义 + 问答记录 + 我讲错过的地方 + 自测题 |

每学完或学了一部分，就更新下面的进度表和学习日志。

## 进度表

状态：✅ 学完（"阶段一" = 这一轮够深了，以后还会回来深入）· 🟡 讲完、还在消化 · ⬜ 没开始

| # | 课 | kernel | 页面 | 笔记 | 状态 | 日期 | FP32 TFLOPS（N=4096） | 一句话收获 |
|---|---|---|---|---|---|---|---:|---|
| 1 | naive | `matmul_naive.cu` | [html](matmul_naive.html) | [notes](matmul_naive.notes.md) | ✅ | 2026-10-01 | 0.97（16×16） | 性能由 warp 视角决定；新 naive 慢在 L1 每条请求碰太多 cache line，不是 HBM |
| 2 | coalesced | `matmul_coalesced.cu` | [html](matmul_coalesced.html) | [notes](matmul_coalesced.notes.md) | ✅ | 2026-10-01 | 5.73 | coalescing 看 32 个 lane 的地址；访存相同不等于性能相同（codegen 差 ~7%） |
| 3 | smem tiling | `matmul_smem.cu` | [html](matmul_smem.html) | [notes](matmul_smem.notes.md) | ✅ | 2026-10-02 | 8.99 | block 共用 smem tile；两个 `__syncthreads` 防 RAW / WAR；瓶颈转到 smem |
| 4 | 1D blocktile | `matmul_1d_blocktile.cu` | [html](matmul_1d_blocktile.html) | [notes](matmul_1d_blocktile.notes.md) | ✅ | 2026-10-02 | 17.56 | register tiling：B 读一次用 8 次；寄存器是 thread 私有的 |
| 5 | 2D blocktile | `matmul_2d_blocktile.cu` | [html](matmul_2d_blocktile.html) | [notes](matmul_2d_blocktile.notes.md) | ✅ 阶段一 | 2026-10-02 ~ 10-03 | 22.21 | 外积：A、B 都进寄存器；SASS 显示编译器把 smem 读合并成 LDS.128，源码字面 ≠ 实际指令 |
| 6 | vectorized | `matmul_vectorized.cu` | — | — | ⬜ | | 32.73 | |
| 10 | warptile | `matmul_warptile.cu` | — | — | ⬜ | | 28.13 | |
| 12 | warptile + 双缓冲 | `matmul_warptile_dbuf.cu` | — | — | ⬜ | | 37.60（调参） | |
| 9a | WMMA（第一个 Tensor Core） | `matmul_wmma.cu` | — | — | ⬜ | | FP16 27.55 | |
| 9c | WGMMA v2 → v9_x | `matmul_wgmma_v*.cu` | — | — | ⬜ | | FP16 104 → ~708 | |

- 编号沿用 `RESULTS.md` 的行号（所以有跳号）。数字来源都是 `RESULTS.md`。
- cuBLAS FP32 是 51.93T，FP16 是 728.7T（end-to-end；kernel-only 数字见 `RESULTS.md` 顶部）。

## 学到的概念

- [x] thread / block / grid；warp = 拉平后连续 32 个 thread（第 1 课）
- [x] row-major；sector（32B）和 cache line（128B）；coalescing（第 1–2 课）
- [x] L1 wavefront、sectors/request；ncu 的基本用法和几个 stall（第 1–2 课）
- [x] SASS 是什么，怎么对比两个 kernel 的指令（第 2 课）
- [x] 测量方法：ABAB 交替、同节点比较、看 CV（第 2 课）
- [x] shared memory tiling；`__syncthreads` 防 RAW / WAR（第 3 课）
- [x] broadcast（跨 lane 复用）vs register reuse（跨指令复用）（第 4 课）
- [x] register tiling；局部数组什么时候进寄存器；local memory / spill（第 4 课）
- [x] 驻留 vs 发射；occupancy 和寄存器的关系（第 4 课）
- [x] BM / BN / BK / TM / TN 命名法（第 4 课）
- [x] 外积；warp / block 在 C 里的形状（第 5 课）
- [x] 跨 block 复用、L2 吸收重复读、block tile 大小和 global 流量的关系（第 5 课）
- [x] smem bank conflict 的数法；`LDS.128` 按半个 warp 一组处理（第 5 课，阶段一）
- [x] wavefront 计数 vs 指令计数；用 1D 当对照组校准 ncu 指标（第 5 课，阶段一）
- [x] 第一次用 SASS 检验推测：发现编译器把 As/Bs 合并成 `LDS.128`，推翻了按源码字面的 bank conflict 分析（第 5 课，2026-10-03）
- [ ] 自己动手：改参数、看 SASS、跑 ncu、验证推测（SASS 是 kernel op 代跑的，还没亲手做过）

## 每课记录：学了什么、纠正了什么、下次能再深入什么

每课四块："覆盖了"是这一轮学到的内容；"关键发现 / 问答"是讨论里最有价值的部分；"纠正过的理解"包括你的和我的；"下次再深入"是重学这一课时可以挖的新东西（阶段二）。完整内容在各课的 `.notes.md`。

### 第 1 课 naive（2026-10-01）

**覆盖了**

- CPU / GPU 分工、kernel、thread / block / grid；warp 由拉平后的编号 `x + y*blockDim.x` 决定。
- row-major 存储；32B sector、128B cache line、coalescing；L1 / L2 / HBM 与数据重用。
- 单个 thread 视角 vs 整个 warp 视角；第一次用 ncu（wavefront、sectors/request、`lg_throttle`）。

**关键发现 / 问答**

- 仓库里旧的 naive 其实是 coalesced 的（x→col）。改成真正的 uncoalesced（x→row，`9b2fb4d`）后：16×16 是 0.97T，32×32 是 0.50T。
- 新 naive 慢，不是因为 HBM 字节多（L1 命中率 99.2%，把字节吸收了），而是 L1 每条请求要处理太多 cache line：sectors/request 16.5，L1 数据通路约 98% 满。
- 32×32 实测比 coalesced 慢 11.44 倍，比"单看 sector 数 6.6 倍"的预期更糟。

**纠正过的理解**

- 我：说 naive 默认 32×32（实际默认 `-b 16`）；把 siboehm kernel 1 的 A / B 说反（实际 A strided、B broadcast）；预期"慢不到 6.6 倍"（实际 11.44 倍）。

**下次再深入**

- 自己用 ncu 看 wavefront 计数（当时那个指标返回 NA，wavefront 模型是推测）。
- 不同 block 形状下 warp 在 C 上的形状，自己推一遍再测。

### 第 2 课 coalesced（2026-10-01）

**覆盖了**

- warp 拉平（复习）；同一 warp 内重复地址只取一次；block 层面的 L1 重用。
- SASS 基础（`cuobjdump -sass`），几种指令和 stall（`long_scoreboard` 等）。
- 测量方法：10 轮 ABAB 交替、同节点比较、看 CV；ncu 要在 `kdev-profiling` pod 上跑（Slurm 节点没权限读计数器）。

**关键发现 / 问答**

- coalesced 和旧 naive 32×32 的访存模式完全一样，但同节点实测慢 6.8%。SASS 指令数差不多（coalesced 还少 3 条），推测是 load 排列顺序不同、在路上的 load 更少。
- 旧 naive 32×32 比 16×16 快 15.9%：load 指令一样，差别在 block 层面的 L1 重用（命中率 95% 对 87.5%）。
- 默认 16×16 是早期 GPU（每 block 最多 512 thread、按 half-warp 合并访存）留下的习惯。

**纠正过的理解**

- 你：猜 16×16 是 FP64 时代的最优值（不是）。
- 我："coalesced 和 naive -b32 性能等价、差距是测量误差"（错，ABAB 证明慢 6.8%）；"差别来自指令数"（SASS 显示不是）；block 层面 sector 估算绝对值偏小约 1.76 倍。

**下次再深入**

- 6.8% 的差距 ncu 下只剩 2%，没完全解释；可以再看 SASS 里第一个 FFMA 之前有多少 load 在路上。

### 第 3 课 smem tiling（2026-10-02）

**覆盖了**

- shared memory：block 共享、程序员管理的片上存储；tiling 让每个 global 元素被 32 个 thread 共用。
- 两个 `__syncthreads`：一个防 RAW（写完才读），一个防 WAR（读完才覆盖）。
- smem bank 的第一次介绍；`#pragma unroll`。

**关键发现 / 问答**

- global load 少了 32 倍，但只快了 1.59 倍：瓶颈转到 smem，每次 FMA 仍要读 2 次 smem。
- warp 视角：`As[ty][k]` 是 broadcast，`Bs[k][tx]` 是连续一行，都没有 bank conflict。
- 学 GPU kernel 的建议：先预测再测量；用三问看访存；记几个硬件数字会算账；读 SASS；把 ncu 当裁判；故意搞坏它；合上代码自己重写；对所有文档（包括 Claude 的）保持怀疑。

**纠正过的理解**

- 我：说 A 的 broadcast "几乎是白花的"（夸大了；真正的问题是每个值在每个 lane 里只用 1 次）。
- 仓库：`worklog.md:134` 的 bank 推理是错的（结论碰巧对）。

**下次再深入**

- 把映射改成 warp 内 ty 变化，亲手制造 32-way bank conflict 再用 ncu 量出来。
- SMEM_TILE 改 16 / 64 会怎样（64 超过 1024 thread 上限）。

### 第 4 课 1D blocktile（2026-10-02）

**覆盖了**

- 寄存器；register tiling（register blocking / thread coarsening），它是 GEMM 最核心的技巧。
- 统一原则：内存层级的每一层都让数据读一次、用很多次。
- 局部数组什么时候进寄存器（下标编译期可知）；local memory / spill。
- 驻留 vs 发射（每 SM 最多 64 个 warp 驻留，每拍最多 4 个发射）；occupancy 和寄存器的关系。
- BM / BN / BK / TM / TN 命名法；搬运映射和计算映射分开。
- 各层内存的访问规则；bank 按 4B 交错编号，连续读不冲突。

**关键发现 / 问答**

- 8.99T → 17.56T（1.95×），经典阶梯前半段单步最大的提升。省的是 B 的 `LDS`：读一次用 8 次。
- broadcast 是跨 lane 的复用，register reuse 是跨指令的复用；要在寄存器里复用一个值，就得把需要它的计算交给同一个 thread。
- 每次 FMA 的 smem 读 = 1 + 1/TM，A 的那 1 次是 1D 的天花板。调参 (64,64,4,16) 只再快 10%。
- 2026-10-03 SASS 补充：1D 的 As 读其实被编译器合并成了 16 条 `LDS.128`；ncu 显示 1D 被 smem 卡着（mio_throttle 3.23），但没有 bank conflict。

**纠正过的理解**

- 你："减少的是 broadcast 次数"（减少的是 B 的连续读）；"8 个 A 已经在寄存器里"（每次从 smem 读）；"bank 是一段连续地址"（按 4B 交错）；"FP16 相邻 lane 同 bank 是冲突"（同一个 word 不算）；"SM 只能跑 4 个 warp、轮流用寄存器"（驻留 ≠ 发射）；"block 太大会 spill"（spill 只看单 thread）；"spill 直接去 HBM"（经过 L1/L2）。
- 我：1D 调参冠军配置写错（应为 (64,64,4,16)）。

**下次再深入**

- 亲手跑 `-Xptxas -v` 看寄存器数；在 SASS 里找 `LDL/STL`。
- 为什么编译器能把 As 合并成 `LDS.128`（dotIdx 展开后同一行连续）——第 5 课已经解释，可以回头用 1D 的 SASS 验证一遍。

### 第 5 课 2D blocktile（2026-10-02 ~ 10-03，阶段一）

**覆盖了**

- 外积：每个 thread 算 8×8，A、B 都进寄存器，16 次 smem 读换 64 次 FFMA；64 个部分和留在寄存器里。
- 寄存器不分类型（32 位格子）；warp 是 C 里 16 行 × 128 列，block 是 8 个 warp 叠成 128×128。
- 跨 block 复用：block 内流式处理、每个元素只从 global 读一次；跨 block 的重复（每字节约 32 次）由 L2 吸收；总读请求 = N³ × 4B × (1/BM + 1/BN)。
- 搬运循环（每 thread 搬 4 个）；bank conflict 的数法（相邻 lane 的地址差几个 word）；ncu 意义上的 wavefront。
- 第一次走完"预测 → SASS → ncu"的验证流程，并用 1D 当对照组。

**关键发现 / 问答**

- SASS：编译器把 As、Bs 都合并成了 `LDS.128`（每 thread 每 dotIdx 4 条，不是 16 条）；寄存器 162 个，每 SM 只放 1 个 block。Simon 在 A6000 上的 As 还是 32 位读，我们的 build 不一样。
- ncu：`LDS.128` 按半个 warp 一组处理；As 读无冲突；Bs 读有真冲突（8 个 wavefront，理想 4 个），但 smem stall ≈0，不是瓶颈。2D 每 SM 只有 8 个 warp（12.5%）。
- Bs 冲突可以通过换列分配（thread c 负责 `c*4+0..3` 和 `64+c*4+0..3`）或 swizzle 消掉，但预计几乎不变快——先修瓶颈。
- bank 是"便宜的并行"换来的代价；AMD、Intel、Apple GPU 都有；TPU 靠编译器、Hopper 靠 TMA + swizzle 避开。NVIDIA ncu 的 wavefront ≠ AMD 的 wavefront（后者相当于 warp）。
- 同样的代码只改参数快 52%（(128,128,16,16,8) 34T）；默认值来自 Simon 的 A6000。
- 这一课在整条阶梯里是中级；最稀缺的技能在后面的 Tensor Core 部分。

**纠正过的理解**

- 你：warp 是 8×256 或 64×128（实际 16×128）；要把整条 A、B 装进 cache（不需要，流式处理）；能装进 L1（装不下，L1 在这里几乎没用）。
- 我：按源码字面推出 As 2-way、Bs 4-way、smem 是瓶颈、上限 22T（SASS + ncu 推翻：As 无冲突，Bs 冲突只有 2 倍且不是瓶颈）；寄存器 100–128、每 SM 2 个 block（实际 162 个、1 个）；68.7% 的出处（实际是 Simon 的 Kernel 5 在 A6000、未调参）；第一次讲得太密。

**下次再深入（阶段二）**

- 2D 的 stall 分布（报告已有，不用重跑 GPU）：卡在 `long_scoreboard` 还是 `barrier`？
- 为什么 TM=16、BK=16 快这么多；TM/TN 不对称的真实原因。
- 为什么我们的 H100 build 把 As 合并成 `LDS.128` 而 Simon 的没有。
- 亲手写换列分配的版本消掉 Bs 冲突，验证"几乎不变快"。
- 写回 C（32 个 lane 落在 32 个 sector）的代价。

## 待验证的推测（需要交给 ops session 实测）

| 来自 | 推测 | 怎么验证 | 状态 |
|---|---|---|---|
| 第 4 课 | 1D 的 As 读被编译器合并成 `LDS.128` | SASS | ✅ 对：16 条 LDS.128（2026-10-03） |
| 第 5 课 | 2D 计算阶段 As 读 2-way、Bs 读 4-way bank conflict | SASS + ncu | 部分对：As 无冲突（LDS.128 broadcast）；Bs 有真冲突（2 倍 wavefront），但 smem stall ≈0，不是瓶颈 |
| 第 5 课 | 2D 用 100–128 个寄存器、每 SM 2 个 block | SASS `-res-usage` | ❌ 实际 162 个、每 SM 1 个 block |
| 第 5 课 | 新嫌疑：每 SM 只有 1 个 block，搬运延迟没被盖住 | ncu occupancy + stall 原因 | occupancy ✅ 实测 8 warp/SM（12.5%）；主要 stall 是哪个还没看 |
| 第 5 课 | HBM 只读约 0.6GB，L2 服务约 4.3GB | ncu `dram__bytes_read.sum` 和 L2 流量 | 没做 |
| 第 5 课 | 2D 瓶颈：smem 读（我，已基本推翻）vs GMEM 指令数（worklog）vs occupancy（新嫌疑） | 上面几项一起看 | 没做 |

## 仓库文档里发现的问题

每课笔记里都有一节"仓库文档里的问题（未修改）"。目前都**只记录、没改**；要改的话需要单独决定。

| 课 | 条数 | 位置 |
|---|---:|---|
| 1 naive | 见笔记 | `matmul_naive.notes.md` |
| 2 coalesced | 见笔记 | `matmul_coalesced.notes.md` 第 8 节 |
| 3 smem | 见笔记 | `matmul_smem.notes.md` |
| 4 1D | 6 | `matmul_1d_blocktile.notes.md` 第 10 节 |
| 5 2D | 11 | `matmul_2d_blocktile.notes.md` 第 9 节 |

## 学习日志

- **2026-10-01**：第 1 课 naive、第 2 课 coalesced。发现旧 naive 本来就是 coalesced 的，改成真正 uncoalesced（`9b2fb4d`）后重测；第一次用 ncu（在 `kdev-profiling` pod 上）。
- **2026-10-02**：
  - 第 3 课 smem tiling、第 4 课 1D blocktile（第 4 课另有一轮很长的问答）。
  - 第 5 课 2D blocktile：第一次讲得太密，又是凌晨 2 点，没读懂。之后改成一次只讲一小块：8×8 外积 → L2 / HBM 复用 → 搬运循环 → bank conflict。bank conflict 还在消化。
- **2026-10-03**：核实第 5 课的 bank conflict。Simon 的博客没讲 2D 的 bank conflict。让 kernel op 跑了 SASS：As、Bs 都被合并成 `LDS.128`，寄存器 162 个、每 SM 1 个 block。按源码字面的 2-way / 4-way 分析不适用于实际指令；新嫌疑是 occupancy。同日 ncu：As 无冲突、Bs 有真冲突但 smem stall ≈0；2D 只有 8 warp/SM。学到 `LDS.128` 按半个 warp 一组处理。第 5 课阶段一收尾，阶段二清单记在笔记里。

## 学习方法（对我有用的）

- 每次只学一小块，讲完先说"懂了"再往下；白天精神好的时候学。
- 每个访存都从两个视角看：单个 thread（代码字面意思）和整个 warp（32 个 lane 的地址）。三问：(1) 每个 lane 的下标是多少；(2) 32 个地址长什么样（相同 / 连续 / 跨步）；(3) 每个值用几次。
- 没被 SASS / ncu / 实测验证的结论都标"推测"。仓库文档不一定对，要批判地看。
- 一条贯穿始终的原则：**在内存层级的每一层，都让数据读一次、用很多次。**

## 下一步

1. 第 6 课 vectorized：先看 SASS。我们的 build 在第 5 课已经把 As 编成 `LDS.128`，所以 22T → 32.7T 的提升从哪来要实测（推测：global 读改 `float4`，或寄存器 / occupancy 变化）。
2. 第 5 课阶段二（以后）：清单见 `matmul_2d_blocktile.notes.md` 第 7 节末尾。
