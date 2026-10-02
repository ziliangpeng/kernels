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
| 1D 自动调参（128,128,4,32） | 19.36 | 37.3% | +10% |

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

## 9. 我说错过 / 纠正过的地方

- 上一课我说 A 的 broadcast "几乎是白花的"。纠正：broadcast 出来的值 32 个 lane 都用了，换来 32 次 FMA，不是浪费。真正的问题是**每个从 smem 读出来的值在每个 lane 里只用 1 次**——本课改的正是这个。

## 10. 仓库文档里的问题（未修改）

- `matmul_1d_blocktile.h`："Expected ~36.5% of cuBLAS (2.8x over smem)" 是 siboehm 在 A6000 上的数字；本机实测 33.8%、1.95×。
- 同一文件说提升来自 "reducing thread management overhead"——主因是 smem 读减少和寄存器复用。
- `worklog.md:268` 说 "One A column held in registers"，但代码里进寄存器的是 `tmpB`（B），A 每次从 smem 读。
- `blockDim` 参数仍未使用。

## 11. 要点 / 自测题

- 省掉的是 smem → 寄存器的 `LDS`：B 读一次进 `tmpB`，复用 8 次。
- "每一层都读一次、用多次"用了第二次。
- 搬运和计算可以用不同的 thread 映射，靠 `__syncthreads` 隔开。
- 局部数组只有在下标编译期可知时才会进寄存器。

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
