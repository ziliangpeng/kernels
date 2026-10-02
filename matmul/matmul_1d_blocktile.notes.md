# 第 4 课学习笔记：1D blocktile（register tiling 的第一步）

[可视化页面](matmul_1d_blocktile.html) · 上一课：[smem tiling](matmul_smem.notes.md) · 下一课：[2D blocktile](matmul_2d_blocktile.notes.md)

源码：`matmul_1d_blocktile.cu` / `matmul_1d_blocktile.h` · H100 · FP32 · N=4096

---

## 1. 核心思路

smem kernel 的问题：每次 FMA 要读 2 次 smem（一个 A、一个 B），**从 smem 读出来的值在每个 lane 里只用 1 次就扔掉**。瓶颈从 global memory 转到了 smem。

1D blocktile：**每个 thread 算 C 同一列上连续 8 个元素**（`TM = 8`）。这 8 个输出在同一个 k 上乘的是同一个 B 值：

```
C[r+0][c] += A[r+0][k] * B[k][c]
C[r+1][c] += A[r+1][k] * B[k][c]
...
C[r+7][c] += A[r+7][k] * B[k][c]   ← 8 行都乘同一个 B[k][c]
```

所以 `B[k][c]` 只从 smem 读一次，放进寄存器 `tmpB`，连续用 8 次。

## 2. 基础概念

### 寄存器

- 每个 thread 私有、最快的存储。读寄存器不需要任何访存指令。
- 到这一课，内存层级三层都用上了：

| 层级 | 存什么 | 谁共享 | 引入于 |
|---|---|---|---|
| global（HBM / L2） | 整个 A、B、C | 所有 block | 一开始 |
| smem | A、B 的 tile | 一个 block | 第 3 课 |
| 寄存器 | `tmpB` 和 8 个累加值 | 一个 thread | 本课 |

### register tiling（register blocking / thread coarsening）

- 让一个 thread 算一小块输出，把复用的值和累加值放在寄存器里。
- **这不是冷门技巧，而是 GEMM 最核心的技巧**：cuBLAS / CUTLASS 的每个高性能 kernel、CPU BLAS（GotoBLAS、OpenBLAS、MKL）的 microkernel 都靠它；tensor core 的 `mma` / `wgmma` 本质上就是硬件完成的"寄存器里的小块外积"。
- 只有"1D"（只在一个方向复用）这个具体形式是教学用的过渡台阶，下一课 2D 才是标准形式。

### 统一原则

**在内存层级的每一层，都让数据读一次、用很多次。**

```
HBM / L2  ──(smem tiling，第 3 课)──▶  smem    一次搬进来，block 内共用
smem      ──(register tiling，本课)──▶ 寄存器  一次读进来，thread 内多次 FMA 共用
寄存器    ──▶ FFMA
```

## 3. 参数

```cuda
#define BM_1D 64   // 一个 block 负责 C 的 64 行
#define BN_1D 64   // 64 列
#define BK_1D 8    // 每轮沿 K 搬 8 列 A、8 行 B
#define TM_1D 8    // 每个 thread 算 8 个输出
#define NUM_THREADS_1D ((BM_1D / TM_1D) * BN_1D)   // 64/8 × 64 = 512
```

- 一个 block 算 64×64 = 4096 个输出，每个 thread 8 个 → 512 个 thread。
- `As` 64×8 = 512 格，`Bs` 8×64 = 512 格 → **每个 thread 正好搬 1 个 A、1 个 B**。BK 取 8 就是为了让这个数对上。
- smem 只用 (512+512) × 4B = 4KB，比上一课 8KB 还少。

## 4. 逐行代码讲解

### 4.1 smem 与 thread 分工

```cuda
__shared__ float As[BM_1D][BK_1D];   // As[64][8]
__shared__ float Bs[BK_1D][BN_1D];   // Bs[8][64]

const int threadCol = threadIdx.x % BN_1D;   // 0..63
const int threadRow = threadIdx.x / BN_1D;   // 0..7
```

- block 是**一维**的（`threadIdx.x` 0..511），需要二维坐标自己用 `%`、`/` 算。
- thread `tid` 负责 C tile 第 `threadCol` 列、第 `threadRow*8 .. threadRow*8+7` 行。
- warp 视角：32 个连续 tid → `threadRow` 相同，`threadCol` 连续 32 个。

### 4.2 指针挪到 block 起点

```cuda
A += blockRow * BM_1D * N;                    // A 的第 blockRow*64 行
B += blockCol * BN_1D;                        // B 的第 blockCol*64 列
C += blockRow * BM_1D * N + blockCol * BN_1D;
```

- 写法上的变化：之后下标都从 0 开始。循环里 `A += BK_1D; B += BK_1D * N;` 让 A 每轮右移 8 列、B 下移 8 行。

### 4.3 累加器在寄存器里

```cuda
float threadResults[TM_1D] = {0.0f};
```

- 循环完全展开、下标编译期可知 → 编译器放进 8 个寄存器。
- **坑**：如果下标运行时才能确定，数组会被放进 local memory（实际在 global memory），就慢了。

### 4.4 搬运阶段

```cuda
const int innerRowA = threadIdx.x / BK_1D;   // 0..63
const int innerColA = threadIdx.x % BK_1D;   // 0..7
const int innerRowB = threadIdx.x / BN_1D;   // 0..7
const int innerColB = threadIdx.x % BN_1D;   // 0..63

As[innerRowA][innerColA] = A[innerRowA * N + innerColA];
Bs[innerRowB][innerColB] = B[innerRowB * N + innerColB];
```

- **搬运和计算用两套不同的映射**：搬运怎么方便访存就怎么分，计算怎么方便复用就怎么分。中间靠 `__syncthreads()` 隔开，互不影响。
- warp 的访存：
  - A：32 个 lane 覆盖 4 行 × 8 列，每行 global 连续 32B = 1 个 sector，4 个 sector 都用满，但分散在 4 条 cache line 上（按 naive 那课 ncu 的结论，L1 要多处理几次 wavefront；但每 8 个 k 才一次，占比不大）。
  - B：同一行连续 32 个 float = 128B，coalesced。
  - 写 smem：地址都等于 `tid`，连续，无 bank conflict。

### 4.5 计算阶段（重点）

```cuda
for (int dotIdx = 0; dotIdx < BK_1D; dotIdx++) {          // 8 个 k
    float tmpB = Bs[dotIdx][threadCol];                    // 读 1 次 B → 寄存器
    for (int resIdx = 0; resIdx < TM_1D; resIdx++)        // 8 个输出
        threadResults[resIdx] +=
            As[threadRow * TM_1D + resIdx][dotIdx] * tmpB; // 读 1 次 A，做 1 次 FMA
}
```

### 4.6 写回

```cuda
C[(threadRow * TM_1D + resIdx) * N + threadCol] = threadResults[resIdx];
```

- 8 次写，每次 warp 写同一行连续 32 个 float，coalesced。

## 5. 访存分析：单个 thread 视角 vs 整个 warp 视角

**单个 thread**，每个 `dotIdx`：

- 读 1 次 `Bs` → `tmpB`。
- 读 8 次 `As`（第 `dotIdx` 列中属于自己的 8 行）。
- 做 8 次 FMA。

**整个 warp**（threadRow 相同，threadCol 连续 32 个）：

| 读 | 32 个 lane 的地址 | 结果 |
|---|---|---|
| `Bs[dotIdx][threadCol]` | 连续 32 个 | 1 次访问，无冲突 |
| `As[threadRow*8 + resIdx][dotIdx]` | 全部相同（threadRow 相同） | broadcast，每个 resIdx 1 次 |

→ 每个 dotIdx：9 次 smem 访问换 8 × 32 = 256 次 FMA。

## 6. 算账

| | smem kernel | 1D blocktile |
|---|---|---|
| 每个输出每个 k 的 smem 读 | 2 | 9/8 |
| 每 256 次 warp-FMA 的 smem 访问 | 16 | 9 |
| 每搬 1 个 float 进 smem 能做的 FMA（BM·BN/(BM+BN)） | 16 | 32 |

- smem 访问少约 1.78 倍。省下的不只是带宽，还有**指令发射槽**：少发一条 `LDS`，就多出一个位置给 `FFMA`。
- block tile 从 32×32 变 64×64，global 访问也减半：tile 越大，同一个 A 被越多列共用、同一个 B 被越多行共用。
- 编译器可能把固定 `resIdx` 时 `As[r][0..7]` 的 8 次读合并成 2 条 `LDS.128`，那样 smem 访问会更少。**未看 SASS，推测。**

为什么只到 17.6T（FP32 峰值 ~67T）——**推测，未测**：

- 按 smem 访问估的上限比 17.6T 高不少，smem 可能已不是唯一瓶颈。
- BK=8：每 8 个 k 两次 `__syncthreads`，两次 sync 之间每个 thread 只有 64 次 FMA。
- 搬运和计算串行，global 延迟没被隐藏（后面 double buffering 解决）。
- A 仍是 8 次读喂 8 次 FMA（下一课 2D 让 A、B 都进寄存器复用）。

## 7. 实测

| kernel（FP32，N=4096，H100） | TFLOPS | 占 cuBLAS（51.93T） | 对上一级 |
|---|---|---|---|
| smem tiling | 8.99 | 17.3% | — |
| 1D blocktile（64,64,8,8） | 17.56 | 33.8% | 1.95× |
| 1D 自动调参（64,64,4,16） | 19.36 | 37.3% | +10% |

- 19.36T 对应的配置以 `1d-autotune-2026-09-26.md` Addendum（job 219258/219259）为准：(64,64,4,16)。之前这里写成 (128,128,4,32)，见第 9 节。未核对原始 CSV。

经典阶梯前半段每一步的倍数：

| 这一步 | 倍数 |
|---|---|
| naive → coalesced | ~1.1×（旧 naive 本来就 coalesced；新 naive 下是 5.9×） |
| coalesced → smem | 1.59× |
| **smem → 1D blocktile** | **1.95×**（前半段单步最大） |
| 1D → 2D blocktile | 1.26× |
| 2D → vectorized | 1.47× |

## 8. 问答记录

**Q：这省的是 smem → 寄存器的 load 吗？**

A：是。省的就是 `LDS` 指令：每次 FMA 从读 2 次 smem 降到 9/8 次。带宽和指令发射槽都省了。

**Q：提升有多大？**

A：8.99T → 17.56T，1.95×，占 cuBLAS 从 17.3% 到 33.8%。是前半段单步最大的提升。

**Q：感觉这很冷门？是常见技巧还是冷门技巧？**

A：正好相反，register tiling 是 GEMM 最核心、最通用的技巧（见第 2 节）。冷门的只是"1D"这个过渡形式和 `BM=64, BK=8` 这些调参数字。从第 3 课到本课是同一个原则用了两次，后面直到 wgmma 都在更极端地重复它。

### 2026-10-02 第二轮讨论

**Q：本课是不是"减少了 broadcast 的次数"？**

A：不是。两个 kernel 里 broadcast 的都是 A，B 都是连续读。固定一个 k、看 C 的 8×32 区域（256 次 FMA）：

| | A（broadcast） | B（连续读） | 合计 |
|---|---|---|---|
| smem kernel | 8 | 8 | 16 |
| 1D blocktile | 8 | **1** | 9 |

减少的是 B 的连续读。两种复用是不同维度：

- broadcast：同一条指令里，跨 32 个 lane（空间上），1 次访问喂 32 个 thread。
- register reuse：同一个 thread 里，跨多条 FMA 指令（时间上），1 次访问喂 8 次 FMA。本课新增的是这个。

**Q：B 的读取怎么从 8 次变成 1 次？**

A：靠**重新分工**，读 smem 的指令本身没变。`B[k][c]` 被 `C[0..7][c]` 这 8 个输出需要：

- smem kernel：这 8 个输出属于 8 个 thread（还在 8 个不同的 warp 里）。**寄存器是 thread 私有的**，别人读进寄存器的值你用不了，所以读 8 次。
- 1D：这 8 个输出归同一个 thread，读一次进 `tmpB`，自己用 8 次。

一句话：要让一个值在寄存器里被复用，就必须把需要它的计算交给同一个 thread。代价是每个 thread 要 8 个累加器，thread 数变少。

**Q：一句话总结 smem 读的账？**

A：每次 FMA：B 被 8 次 FMA 分摊，只要 1/8 次；A 仍要完整读 1 次，合计 1.125 次（smem kernel 是 2）。这 8 个 A 是每次 FMA 当场从 smem 读的（按顺序 8 条 LDS），不是一次读完。

**Q：为什么 TM 停在 8？用 16、32 不是更好吗？**

A：可以，但收益递减。每次 FMA 的 smem 访问 = 1 + 1/TM：

| TM | 每次 FMA 的 smem 访问 |
|---|---|
| 1（smem kernel） | 2 |
| 8 | 1.125 |
| 16 | 1.0625 |
| 32 | 1.03 |

A 的那 1 次省不掉，所以永远不会低于 1。代价随 TM 增大：本课"每人正好搬 1 个 A、1 个 B"的写法要求 BN = BK × TM。BM = BN = 64 时，TM=16 → BK=4、256 个 thread；TM=32 → BK=2、128 个 thread。也就是 BK 越来越薄、sync 越来越频繁、thread 越来越少、累加器越来越多。

实测（`1d-autotune-2026-09-26.md` Addendum）：(64,64,8,8) 17.56T → (64,64,4,16) 19.36T，+10%，确实更好，但远小于 smem → 1D 的 1.95×。默认的 8 来自 siboehm 教程，最优值靠 autotune 实测。要突破"每次 FMA 读 1 次 A"，只能让 A 也进寄存器复用，这就是第 5 课的 2D。

**Q：BM、BN、BK、TM、TN 这些名字怎么记？**

A：C[M×N] = A[M×K] × B[K×N]。名字 = 层级字母 + 维度字母：

- 层级：**B** = block，**W** = warp（warptile 那课用到），**T** = thread。
- 维度：**M** = 行，**N** = 列，**K** = 累加方向。
- 例如 BK = 一个 block 每轮沿 K 搬多厚；TM = 一个 thread 算几行。本课 TN = 1 所以没写。没有 TK，因为 thread 在 K 方向就是逐个循环。
- 每个 block 的 thread 数 = (BM/TM) × (BN/TN)。
- 通用写法：siboehm 教程用同样的名字；Triton 是 `BLOCK_SIZE_M/N/K`；CUTLASS 是 `ThreadblockShape<M,N,K>`、`WarpShape<M,N,K>`。

**Q：BK 通常等于 TM 吗？**

A：不是。BK 管 K 方向（smem 每轮多厚、两次 sync 之间循环几次），TM 管 M 方向（B 复用几次）。本课相等是因为约束 BN = BK × TM = 64 恰好拆成 8 × 8。反例：(64,64,4,16)、(64,64,16,4)、(128,128,2,64)。后面几课用循环或向量化搬运，这个约束本身也不存在了。

**Q：1D 只是教学用的？第 5 课才是大家实际的做法？**

A：要分开说：

- register tiling 的思想到处都在用；只有"只复用一个方向"的 1D 形式是过渡台阶。
- 第 5 课的 2D thread tile（每个 thread 算 TM×TN 外积）是 CUDA core GEMM 的标准结构（CUTLASS SIMT kernel、经典手写 sgemm）。
- 但生产 kernel 还会叠加：向量化读、warp 分块、双缓冲 / `cp.async` 流水线、smem swizzle。仓库阶梯到 dbuf 才 37.6T（cuBLAS 的 72%）。
- H100 上 FP16/BF16 实际用 tensor core（`wgmma`），"寄存器里的小块外积"由硬件完成。
- 推测：cuBLAS FP32 基准 51.93T 低于 FFMA 峰值 ~67T，所以应该是 FFMA SIMT kernel，不是 TF32 tensor core。未用 ncu 确认。

**Q：每个 thread 读的 8 个 A 在 smem 里是跳着的（间隔 8 个 float），不是问题吗？是不是已经在寄存器里，可以随便读？**

A：这 8 个 A 不在寄存器里，每次 FMA 当场从 smem 读。跳着读不是问题，因为**访存好坏只看同一条指令里 32 个 lane 的地址，不看一个 thread 先后读的地址**。`LDS As[r][k]` 这条指令 32 个 lane 地址相同 → broadcast；下一条跳 8 个 float，是另一条指令，硬件不会合在一起评判。

寄存器也不能"随便读"：寄存器**没有地址**，只能用编译期确定的下标访问。

"一个 thread 先后读的地址连续"只在**向量化**时有用：`As[r][0..7]`（同一行、不同 k）连续，才有"可能合并成 `LDS.128`"的推测；`As[r+0..7][k]` 不连续，不能合并。

补充：搬运阶段 `As[...] = A[...]` 实际是 `LDG` 进中转寄存器 → `STS` 写进 smem（后面的 `cp.async` 可以跳过这个寄存器）。

**Q：各层内存的访问规则？**

A：global 和 smem 的规则都按"一个 warp 的一条指令、32 个 lane 的地址"评判。容量是 H100 规格书数字，未实测：

| 层级 | 大小 | 粒度 / 规则 | 好 | 坏 |
|---|---|---|---|---|
| HBM / global | 80GB，~3.35TB/s | 32B sector，4 个 sector = 128B line | 32 个 lane 连续 32 个 float = 4 个 sector（coalesced） | 大跨步 → 每个 lane 单独占 1 个 sector |
| L2 | 50MB，全部 SM 共享 | sector 粒度 | 多个 block 重复读的数据命中 L2 | — |
| L1 | 和 smem 共用每 SM 256KB | 128B line / 32B sector，看 wavefront 数 | 一条指令的地址集中在少数几条 line | 分散在多条 line（第 1 课 naive 的问题） |
| smem | 每 SM 最多 ~228KB | 32 个 bank × 4B，bank = (地址/4) % 32 | 32 个不同 bank，或全部同地址（broadcast） | 不同地址落在同一个 bank → n-way conflict |
| 寄存器 | 每 SM 64K 个，每 thread ≤255 | 没有地址 | 下标编译期固定 | 运行时下标 → local memory；用太多 → occupancy 下降甚至 launch 失败 |

本课四种访问：搬 A 是 4 行 × 每行 8 个 float（4 个 sector 用满但在 4 条 line，稍差）；搬 B 连续 32 个（coalesced）；读 `Bs[k][threadCol]` 连续（无 conflict）；读 `As[r][k]` 同地址（broadcast）。

**Q：smem 和 L1 物理上是同一块，L1 也有 bank conflict 吗？conflict 是 warp 内还是跨 warp？连续读会冲突吗？**

A：

- smem 和 L1 共用同一块 SRAM（Volta 起，H100 每 SM 256KB，可配置划分），但访问路径不同：smem 由程序员直接给地址，看 bank conflict；L1 是硬件缓存，看一条指令碰到几条 line（wavefront）。写代码时 bank conflict 只是 smem 的概念。L1 内部是否也有 bank 冲突、ncu 有没有对应指标——不确定，推测。
- **只在同一个 warp 的同一条指令内**判断。不同 warp 是不同指令、不同时间处理，只会争带宽，不算 bank conflict。
- bank 按 4 字节**交错**编号：第 0..31 个 float 在 bank 0..31，第 32 个回到 bank 0（不是"一段连续地址属于同一个 bank"）。所以**连续读没有冲突**，是最好的模式。
- 冲突 = 不同地址落在同一个 bank。stride 1 → 无；stride 2 → 2-way；stride 8 → 8-way；stride 32（读 `float s[32][32]` 的一列）→ 32-way；声明成 `s[32][33]`（padding）→ 无。奇数 stride 一定无冲突（与 32 互质）。全部同地址是 broadcast，不算冲突。

**Q：FP16 时相邻两个 lane 落在同一个 bank，是冲突吗？所以 FP16 要每人读两个？**

A：相邻两个 lane 确实在同一个 bank，但读的是**同一个 4B word** 的两半，硬件读一次就能满足两个 lane，**不算冲突**。真正的问题是 32 个 lane 只读了 64B，只用了 16 个 bank，带宽浪费一半。所以 FP16 通常每人读 `half2`（4B，一条指令 128B 用满 32 个 bank）或更宽。

推测（基于公开资料，未在 H100 验证）：单 lane 超过 4B（`LDS.64`/`LDS.128`）时总请求超过 128B，硬件把 warp 拆成几组分轮处理，bank conflict 在组内判断。用 `LDS.128` 的意义是用更少的指令读完同样的数据，把发射槽留给 FFMA。

**Q：两套映射、两个 `__syncthreads` 是怎么回事？**

A：同一个 thread 在两个阶段负责的东西不同。以 tid=100 为例：搬 `As[12][4]`、`Bs[1][36]`；计算 C 第 8..15 行、第 36 列。它算的时候要读 `As[8..15][*]`，但它自己只搬了其中一格，其余都是别人搬的。

- 第一个 sync：等所有人都写完 smem，才能读。
- 第二个 sync（循环末尾）：等所有人都读完，才允许下一轮覆盖。否则跑得快的 thread 会覆盖跑得慢的 thread 还在读的数据。

分两套映射是因为目标不同：搬运要 coalesced、无 bank conflict；计算要复用。推测：每 8 个 k 两次全 block 同步，同步时没人在做 FFMA，是 17.6T 的瓶颈之一；双缓冲可以去掉第二个 sync 的代价。

**Q：累加器为什么能放在寄存器里？**

A：按 C 的规定数组要有地址，寄存器没有地址。所以只有**每次访问的下标都是编译期常数**时，编译器才能把 `threadResults[8]` 拆成 8 个独立的寄存器。本课 `TM_1D` 是 `#define` 常数，加 `#pragma unroll` 展开后下标都是常数（初始化和写回循环也一样）。

否则数组落到 **local memory**："local"只表示每个 thread 私有，物理上在 HBM（经 L1/L2 缓存），每次 `+=` 变成 `LDL + FFMA + STL`。下标全是常数但寄存器用太多，也会 spill 到 local memory。

验证方法：编译加 `-Xptxas -v` 看 `Used N registers` 和 spill 字节数；或在 SASS 里搜 `LDL/STL`。**本课实际用了多少寄存器未测**，我估计二三十到四十个（推测）。

**Q：寄存器不够时为什么不 spill 到 smem？是直接去 HBM 吗？**

A：不是直接去 HBM：local memory 经过 L1/L2，spill 少时大多命中 L1。默认不放 smem，我的理解（推测）：smem 按 block、在 launch 时确定、由程序员掌控，编译器按 thread 偷占会挤占程序员的 smem、降低 occupancy；local memory 需要一块按 thread 私有、几乎无上限的后备空间（还要放调用栈），只有 HBM 够大。记得 CUDA 13.0 加了需要主动开启的 smem spill 功能，未查证。

**Q：block 太大 → warp 太多 → 容易 spill，对吗？SM 是不是只能跑 4 个 warp、warp 轮流用寄存器？**

A：要区分**驻留**和**发射**：

| | H100 每 SM | 含义 |
|---|---|---|
| 驻留 | 最多 64 个 warp（2048 thread） | 同时占着寄存器，各有各的，不轮流 |
| 发射 | 每周期最多 4 个 warp | 4 个 partition，每个 1 个 scheduler，每周期挑 1 个 warp 发 1 条指令 |

轮流的是发射，不是寄存器。所有驻留 warp 的寄存器一直在寄存器堆里，所以切换 warp 没有代价——一个 warp 等访存时，下个周期就换另一个，这是 GPU 隐藏延迟的核心。寄存器堆 64K 分给 4 个 partition，每个 16K。

spill 是编译时、按单个 thread 决定的（需要的寄存器是否超过上限，默认 255，可被 `__launch_bounds__` / `-maxrregcount` 压低）。block 大小 × 每 thread 寄存器影响的是 occupancy，以及一个 block 要的寄存器是否超过 SM 的 65536 个（超过就 launch 失败）。只有写了 `__launch_bounds__(512)` 时，编译器才必须把每 thread 压到 128 以内，才可能因此 spill。

占用表（512 thread/block，忽略分配粒度取整）：R ≤ 32 → 4 个 block；R ≤ 64 → 2 个；R ≤ 128 → 1 个。

推测：autotune 里 (256,256,2,128) LAUNCH_FAIL，是 512 thread × 每 thread 128+ 个累加器 > 65536，即一个 block 要的寄存器比整个 SM 还多，而不是 spill。需 `-Xptxas -v` 确认。

**Q：累加器在寄存器里，帮的是读还是写？怎么更新？**

A：都帮，而且在同一条指令里原地完成：`FFMA R13, R20, R21, R13`（R13 = R20 × R21 + R13），不碰内存。每个 thread 8 × 4096 = 32768 次更新全在寄存器里，最后只发 8 条 `STG`。如果落到 local memory，每次更新是 `LDL → FFMA → STL` 三条指令。

## 9. 我说错过 / 纠正过的地方

- 上一课我说 A 的 broadcast "几乎是白花的"。纠正：broadcast 出来的值 32 个 lane 都用了，换来 32 次 FMA，不是浪费。真正的问题是**每个从 smem 读出来的值在每个 lane 里只用 1 次**——本课改的正是这个。
- 我之前把 1D 自动调参写成"(128,128,4,32) 19.36T"。纠正：`1d-autotune-2026-09-26.md` Addendum 里 19.36T 对应 (64,64,4,16)；(128,128,4,32) 是作废的 `-dc` 编译 sweep 里的冠军（13.34T）。像是把两次 sweep 的配置和数字拼到了一起。未核对 CSV。

### 学习时理解偏、已纠正的地方

- "本课减少的是 broadcast 次数" → 减少的是 B 的连续读（8 → 1），A 的 broadcast 次数没变。
- "warp 一次处理 8 个 A" → 每个 thread 按顺序发 8 条 LDS、8 条 FFMA。
- "8 个 A 已经在寄存器里，可以随便读" → A 每次从 smem 读；寄存器没有地址，不能按运行时下标读。
- "bank 是一段连续地址属于同一个 bank" → bank 按 4B 交错编号，连续读正好铺满 32 个 bank。
- "FP16 相邻 lane 同 bank 是冲突" → 同一个 word 不算冲突，问题是只用了一半带宽。
- "SM 只能跑 4 个 warp、warp 轮流用寄存器" → 每周期最多发射 4 个，但最多驻留 64 个，寄存器同时存在。
- "block 太大会导致 spill" → spill 只看单 thread 寄存器需求；block 大影响 occupancy / 能否 launch（除非用了 `__launch_bounds__`）。
- "spill 直接去 HBM" → 经过 L1/L2 缓存，HBM 只是后备存储。

## 10. 仓库文档里的问题（未修改）

- `matmul_1d_blocktile.h`："Expected ~36.5% of cuBLAS (2.8x over smem)" 是 siboehm 在 A6000 上的数字；本机实测 33.8%、1.95×。
- 同一文件说提升来自 "reducing thread management overhead"——主因是 smem 读减少和寄存器复用。
- `worklog.md:268` 说 "One A column held in registers"，但代码里进寄存器的是 `tmpB`（B），A 每次从 smem 读。
- `blockDim` 参数仍未使用。
- `RESULTS.md:23` 写 1D tuned FP32 是 (64,64,4,16)，`RESULTS.md:106` 写成 (128,128,4,32) 19.36T，同一文件前后矛盾；按 `1d-autotune-2026-09-26.md` Addendum 应为 (64,64,4,16)。
- `1d-autotune-2026-09-26.md` 主表是作废的 `-dc` 编译数据（文件开头已声明），但 Findings 里"同一冠军 (128,128,4,32)"等结论没有标注作废，容易误读。

## 11. 要点 / 自测题

- 省掉的是 smem → 寄存器的 `LDS`：B 读一次进 `tmpB`，复用 8 次。
- "每一层都读一次、用多次"用了第二次。
- 搬运和计算可以用不同的 thread 映射，靠 `__syncthreads` 隔开。
- 局部数组只有在下标编译期可知时才会进寄存器。
- 寄存器是 thread 私有的：要复用一个值，就把需要它的计算交给同一个 thread。
- broadcast 是跨 lane 的复用，register reuse 是跨指令的复用；本课 A 仍只有前者。
- 每次 FMA 的 smem 访问 = 1 + 1/TM，A 的那 1 次是 1D 的天花板。
- 访存好坏只看同一条指令里 32 个 lane 的地址，不看一个 thread 先后读的地址。
- 驻留 ≠ 发射：最多 64 个 warp 驻留，每周期最多 4 个发射。

<details><summary>自测 1：为什么 BK 是 8？</summary>

512 个 thread，As = BM×BK = 64×BK、Bs = BK×BN = BK×64。BK=8 时两者都是 512 格，每个 thread 正好搬 1 个 A、1 个 B，搬运代码最简单。
</details>

<details><summary>自测 2：如果改成每个 thread 算一行上的 8 个输出（TN=8，而不是一列 TM=8），哪个矩阵的值进寄存器复用？warp 视角下 smem 访问变成什么样？</summary>

一行上的 8 个输出 `C[r][c..c+7]` 共享同一个 `A[r][k]`，所以复用的是 A（`tmpA`），每次读 8 个 B：`Bs[k][c*8 + j]`。

warp 视角：如果 32 个 lane 的 r 相同、c 连续，那么 `As[r][k]` 是 broadcast；`Bs[k][lane*8 + j]` 的 32 个地址间隔 8 个 float → 只落在 4 个 bank（0、8、16、24 + j）→ **8-way bank conflict**。这就是为什么直接"转个方向"会变慢，需要配合换 smem 布局或向量化读（`LDS.128` 一次读连续 4 个）。
</details>

<details><summary>自测 3：<code>threadResults[resIdx]</code> 如果 <code>resIdx</code> 是运行时才知道的值，会发生什么？</summary>

编译器无法把数组映射到固定寄存器，会放进 local memory（物理上在 global memory，经 L1 缓存），每次累加都变成访存，性能大跌。ncu / SASS 里会看到 `LDL` / `STL`。
</details>

<details><summary>自测 4：把 <code>Bs</code> 转置成 <code>Bs[64][8]</code>，计算阶段读 <code>Bs[threadCol][dotIdx]</code>，是几路 bank conflict？</summary>

32 个 lane 的 `threadCol` 连续，地址 = threadCol × 8 + dotIdx，stride = 8 个 float → 只落在 4 个 bank → **8-way conflict**。
</details>

<details><summary>自测 5：删掉循环末尾的 <code>__syncthreads()</code> 会怎样？</summary>

跑得快的 warp 会进入下一轮、覆盖 `As/Bs`，而慢的 warp 还在读这一轮的数据 → 结果错误。是否出错取决于 warp 的调度时序，所以通常是**偶尔错、不稳定**的 race，比一直错更难调试。
</details>

<details><summary>自测 6：TM 从 8 改成 16，每次 FMA 的 smem 访问从多少变成多少？为什么 TM 不能无限大？</summary>

1.125 → 1.0625。A 每次 FMA 固定读 1 次，收益递减；同时 BK 变薄（sync 更频繁）、thread 变少、累加器变多，太大会寄存器不够、launch 失败。
</details>
