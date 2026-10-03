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
- [ ] smem bank conflict 的分析：已讲，还在消化（第 5 课）
- [ ] wavefront 计数 vs 指令计数；用它估上限：已讲，还在消化（第 5 课）
- [x] 第一次用 SASS 检验推测：发现编译器把 As/Bs 合并成 `LDS.128`，推翻了按源码字面的 bank conflict 分析（第 5 课，2026-10-03）
- [ ] 自己动手：改参数、看 SASS、跑 ncu、验证推测（SASS 是 kernel op 代跑的，还没亲手做过）

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
