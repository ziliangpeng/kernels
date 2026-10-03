# 第 5 课学习笔记：2D blocktile（外积，A 和 B 都进寄存器）

[可视化页面](matmul_2d_blocktile.html) · 上一课：[1D blocktile](matmul_1d_blocktile.notes.md) · 下一课：[vectorized](matmul_vectorized.notes.md) · 总进度：[PROGRESS.md](PROGRESS.md)

源码：`matmul_2d_blocktile.cu` / `matmul_2d_blocktile.h` · H100 · FP32 · N=4096

状态（2026-10-03）：三块内容都讲完了。SASS 已实测（第 5.4 节），**推翻了第 4.3–5.2 节"按源码字面算"的几个结论**：编译器把 As、Bs 都合并成了 `LDS.128`，寄存器 162 个、每 SM 只放 1 个 block。第 4.3–5.2 节保留原样作为"按源码推理"的记录，读的时候以第 5.4 节为准。ncu 还没做。

---

## 1. 核心思路

### 1.1 第 4 课停在哪

1D blocktile 让每个 thread 算 C 里**一列 8 个**输出：

- `tmpB = Bs[dotIdx][threadCol]` 读一次进寄存器，用 8 次；
- A 仍然**每做一次 FMA 就从 smem 读一次**，只是整个 warp 读同一个地址（broadcast）。

所以 B 的复用做到了寄存器层，A 还没有。每次 FMA 的 smem 读 = 1 + 1/TM，A 的那 1 次就是 1D 的天花板。

### 1.2 2D 的做法：一个 thread 算一个 8×8 方块，做外积

每个 thread 负责 C 里一个 **8×8 方块**（`TM = TN = 8`，64 个输出）。每个 `dotIdx`：

```
            b0    b1   ...   b7      ← regB[8]：Bs 第 dotIdx 行里连续 8 个
     a0   a0·b0 a0·b1  ...  a0·b7
     a1   a1·b0 a1·b1  ...  a1·b7
     ..    ...
     a7   a7·b0 a7·b1  ...  a7·b7    ← threadResults[8][8] += ...
     ↑ regA[8]：As 第 dotIdx 列里属于自己的 8 个
```

- 从 smem 读 8 + 8 = 16 个数，做 8 × 8 = 64 次 FMA。
- 每个读进来的数**用 8 次**：`a0` 被一整行用，`b0` 被一整列用。1D 里 B 用 8 次、A 只用 1 次。
- 一句话：**1D 只复用 B，2D 把 A 也复用了。**

### 1.3 相对第 4 课改了什么

| | 1D | 2D | 变了吗 |
|---|---|---|---|
| smem tile + 两个 `__syncthreads` | 有 | 有 | 一样 |
| BK = 8，沿 K 一块块往前走 | 是 | 是 | 一样 |
| 搬运映射和计算映射分开 | 是 | 是 | 一样 |
| ① 每个 thread 的输出形状 | 一列 8 个 | **8×8 方块，64 个** | 改了 |
| ② 内层循环 | B 进寄存器，A 每次读 smem | **A、B 都进寄存器，外积** | 改了 |
| ③ 尺寸 | block 64×64，512 thread，每人搬 1 个 | **block 128×128，256 thread，每人搬 4 个** | 改了 |

③ 是被 ① 逼出来的：如果 block 还是 64×64，每个 thread 算 64 个输出，只需要 64 个 thread，太少。放大到 128×128，就是 128·128 / 64 = 256 个 thread。

### 1.4 参数

```cuda
#define BM_2D 128   // 一个 block 负责 C 的 128 行
#define BN_2D 128   // 128 列
#define BK_2D 8     // 每轮沿 K 搬 8 列 A、8 行 B
#define TM_2D 8     // 每个 thread 算 8 行
#define TN_2D 8     // × 8 列
#define NUM_THREADS_2D ((BM_2D / TM_2D) * (BN_2D / TN_2D))   // 16 × 16 = 256
```

| | 值 |
|---|---|
| smem | As[128][8] + Bs[8][128] = (1024 + 1024) × 4B = 8KB，没有 padding |
| 寄存器 | 源码层 64 + 8 + 8 = 80 个 float；**SASS 实测 162 个、无 spill**（我原来推测 100–128，错了） |
| occupancy | 162 × 256 ≈ 41.5K > 32K → **每个 SM 只放得下 1 个 block = 8 个 warp**（按寄存器算出来的，没用 ncu 测）。我原来推测 2 个，错了 |

## 2. 基础概念

### 2.1 外积（outer product）

两个向量 a（长 m）、b（长 n）的外积是一个 m×n 矩阵，第 (i, j) 格是 `a[i] · b[j]`。矩阵乘法可以写成 K 个外积的和：

```
C = Σ_k  A[:, k] ⊗ B[k, :]      （A 的第 k 列 × B 的第 k 行）
```

2D blocktile 里，每个 thread 对自己的 8×8 方块做这件事：每个 k 加一个 8×8 外积，加 4096 次（512 轮 tile × 每轮 8 个 dotIdx）。

### 2.2 寄存器的"类型"

GPU 寄存器**不分类型**，每个都是 32 位的格子：

- 放 float 就是 float，放 int 就是 int，由指令决定怎么解释（`FFMA` 当 float，`IADD` 当整数）；
- 一个 float 占 1 个寄存器，double 占 2 个；
- 这个 kernel 全是 FP32（`float`）。

### 2.3 部分和留在寄存器里

64 个 `threadResults[i][j]` 是**部分和**：K 方向每走一步就往里累加一次，全部 4096 步走完，才一次性写回 C。中间从不写回内存。

### 2.4 warp 和 block 在 C 里的形状

`threadCol = tid % 16`，`threadRow = tid / 16`。

```
warp 0 = tid 0..31
  tid 0..15  → threadRow = 0, threadCol = 0..15   ← 一排 16 个 8×8 方块
  tid 16..31 → threadRow = 1, threadCol = 0..15   ← 下面再一排

→ 2 排 × 16 个方块 = (2×8) 行 × (16×8) 列 = 16 行 × 128 列
```

- 一个 warp = C 里 **16 行 × 128 列**的一条（矮而宽）。核对：32 个 thread × 64 = 2048 = 16 × 128。
- 一个 block = 8 个 warp 从上到下叠起来 = 128 × 128。
- 一个 block 只在一个 SM 上跑，但一个 SM 可以同时放好几个 block（本 kernel 推测 2 个）。

### 2.5 smem bank 和 bank conflict（本课第一次真正遇到）

- smem 分成 **32 个 bank**，每个宽 4B。地址按 word（4B）编号：**bank = word 地址 % 32**。连续的 32 个 float 正好铺满 32 个 bank。
- 判断单位是**同一个 warp 的同一条指令**里 32 个 lane 给出的地址：

| 情况 | 结果 |
|---|---|
| 32 个地址落在 32 个不同 bank | 1 个 wavefront（1 拍） |
| 多个 lane 读**同一个地址** | broadcast，不算冲突 |
| 多个**不同地址**落在**同一个 bank** | n-way conflict：同一 bank 里有几个不同地址，就拆成几拍 |

- 一个好用的判断法：**同一条指令里，相邻两个 lane 的地址差几个 word？** 差 0 → broadcast；差 1 → 铺满 32 个 bank；差 8、64 这类与 32 有大公因数的数 → 会撞。

### 2.6 wavefront：数"拍"而不是数"指令"

smem 每个 SM 大约每拍处理 1 个 wavefront（推测的近似模型，H100 规格 128B/clk/SM）。一条有 n-way 冲突的 LDS 指令要 n 个 wavefront。所以衡量 smem 压力要数 wavefront，**只数 LDS 指令会高估收益**。

## 3. 逐行代码讲解

### 3.1 smem 声明（第 20–21 行）

```cuda
__shared__ float As[BM_2D][BK_2D];   // As[128][8]，A 块原样存（行优先）
__shared__ float Bs[BK_2D][BN_2D];   // Bs[8][128]
```

没有 padding（`RESULTS.md:82` 说有，是错的，见第 9 节）。

### 3.2 计算用的线程映射（第 24–25 行）

```cuda
const int threadCol = threadIdx.x % (BN_2D / TN_2D);  // 0..15
const int threadRow = threadIdx.x / (BN_2D / TN_2D);  // 0..15
```

- 单 thread：线程 t 负责 C 块第 `threadRow*8 .. +7` 行、第 `threadCol*8 .. +7` 列。例：tid = 17 → threadRow 1、threadCol 1 → 行 8..15、列 8..15。
- warp：threadRow 只有 2 个值，threadCol 0..15 各出现两次（第 2.4 节）。和 1D 不同：1D 是 `threadRow = tid/64`，一个 warp 里 threadRow 全一样。

### 3.3 指针挪到 block 起点（第 32–34 行）

```cuda
A += blockRow * BM_2D * N;                      // A 的第 blockRow*128 行
B += blockCol * BN_2D;                          // B 的第 blockCol*128 列
C += blockRow * BM_2D * N + blockCol * BN_2D;
```

和 1D 一样，只是步长换成 128。

### 3.4 寄存器（第 37–41 行）

```cuda
float threadResults[TM_2D][TN_2D] = {{0.0f}};   // 64 个累加器
float regA[TM_2D];                              // 8 个
float regB[TN_2D];                              // 8 个
```

所有循环都 `#pragma unroll`、下标都是编译期常数，所以编译器能把它们拆成独立寄存器（第 4 课讲过的条件）。

### 3.5 搬运用的映射（第 46–52 行）

```cuda
const int strideA = NUM_THREADS_2D / BK_2D;   // 256 / 8   = 32
const int strideB = NUM_THREADS_2D / BN_2D;   // 256 / 128 = 2
const int innerRowA = threadIdx.x / BK_2D;    // 0..31
const int innerColA = threadIdx.x % BK_2D;    // 0..7
const int innerRowB = threadIdx.x / BN_2D;    // 0..1
const int innerColB = threadIdx.x % BN_2D;    // 0..127
```

- 每轮要装 As 128×8 = 1024 个、Bs 8×128 = 1024 个。256 个 thread → 每人 4 个 A、4 个 B → 4 轮。第 4 课正好每人 1 个，所以那时不需要循环。
- 把 256 个 thread 想成"印章"：搬 A 时印章是 32 行 × 8 列，往下盖 4 次；搬 B 时是 2 行 × 128 列，往下盖 4 次。

### 3.6 搬 A（第 57–64 行）

```cuda
for (int loadOffset = 0; loadOffset < BM_2D; loadOffset += strideA) {   // 0, 32, 64, 96
    int row = innerRowA + loadOffset;
    if (blockRow * BM_2D + row < N && tileIdx + innerColA < N)
        As[row][innerColA] = A[row * N + innerColA];
    else
        As[row][innerColA] = 0.0f;
}
```

- 单 thread：tid = 17 → innerRowA 2、innerColA 1 → 搬 As 第 2、34、66、98 行的第 1 列。tid = 200 → 第 25、57、89、121 行的第 0 列。
- warp：见第 4.2 节。

### 3.7 搬 B（第 67–74 行）

```cuda
for (int loadOffset = 0; loadOffset < BK_2D; loadOffset += strideB) {   // 0, 2, 4, 6
    int row = innerRowB + loadOffset;
    ... Bs[row][innerColB] = B[row * N + innerColB];
}
```

- 单 thread：tid = 17 → Bs 第 0、2、4、6 行的第 17 列。tid = 200 → 第 1、3、5、7 行的第 72 列。

### 3.8 sync 和推进指针（第 76–80 行、第 107 行）

和 1D 一样：第一个 `__syncthreads` 防 RAW（写完才读），循环末尾的第二个防 WAR（读完才覆盖）。`A += 8` 右移 8 列，`B += 8 * N` 下移 8 行。

### 3.9 取 regA（第 87–89 行）

```cuda
for (int i = 0; i < TM_2D; i++)
    regA[i] = As[threadRow * TM_2D + i][dotIdx];
```

- 单 thread：读 As 的**一列**中属于自己的 8 个数。相邻两个相差 8 个 word（跨一整行），所以一个 dotIdx 内不能合并成向量读。
- warp：2-way bank conflict（第 4.3 节）。

### 3.10 取 regB（第 93–95 行）

```cuda
for (int j = 0; j < TN_2D; j++)
    regB[j] = Bs[dotIdx][threadCol * TN_2D + j];
```

- 单 thread：一行里**连续 8 个 float**（32B，起点 32B 对齐）。编译器**有可能**发成 2 条 `LDS.128`（推测，没看 SASS）。
- warp：按标量 LDS 算是 4-way bank conflict（第 4.3 节）。

### 3.11 外积（第 99–104 行）

```cuda
for (int i = 0; i < TM_2D; i++)
    for (int j = 0; j < TN_2D; j++)
        threadResults[i][j] += regA[i] * regB[j];   // 64 条 FFMA，全在寄存器
```

### 3.12 写回 C（第 112–121 行）

```cuda
C[(threadRow * TM_2D + i) * N + threadCol * TN_2D + j] = threadResults[i][j];
```

- 每个 thread 64 次写，整个 kernel 只写这一次。
- warp：固定 (i, j) 时，16 个 threadCol 相差 8 个 float（32B），再乘 2 行 → 32 个 lane 落在 32 个不同 sector，每个只写 4B。模式很差，但主循环有 512 轮，所以占比很小（推测）。
- 每个元素都有边界判断，N 又是运行时值，推测编译器发不出 `STG.128`。

## 4. 访存分析：单个 thread 视角 vs 整个 warp 视角

三问：(1) 每个 lane 的下标是多少；(2) 32 个地址长什么样；(3) 每个值用几次。

### 4.1 单个 thread 视角（代码字面意思）

每个 tile（8 个 dotIdx）：

- 搬运：4 个 A + 4 个 B（global → smem）；
- 计算：每个 dotIdx 读 8 个 As（一列）+ 8 个 Bs（一行连续 8 个），做 64 次 FMA；
- 结束：64 次写 C。

### 4.2 warp 视角：搬运阶段（干净）

以 warp 0、某一轮为例：

| | 32 个 lane 的位置 | global 地址 | 写 smem |
|---|---|---|---|
| A | lane 0–7 在第 0 行、8–15 在第 1 行……共 **4 行 × 8 列** | 4 段，每段连续 8 个 float = 32B = 1 个 sector；**4 个 sector 都用满**，分在 4 条 cache line | 地址 = tid + 常数，连续 → 无冲突 |
| B | 32 个 lane 同一行，列连续 32 个 | **连续 128B = 1 条 cache line**，coalesced | 连续 → 无冲突 |

每个值只搬一次，复用发生在计算阶段。A 一次请求碰 4 条 line，但每个 sector 都用满了，不算浪费。（worklog 说"warp 读 A 的同一行"，是错的，见第 9 节。）

### 4.3 warp 视角：计算阶段（按源码字面推理；已被 SASS 部分推翻，见第 5.4 节）

> 下面的分析假设每个 `As[...]`、`Bs[...]` 都是一条 32 位 `LDS`。SASS 显示实际是 `LDS.128`，所以 2-way / 4-way 的结论**在我们的 build 里不成立**。保留下来，是因为"按 32 位读取怎么数 bank"这套方法本身是对的，下一课也会用到。

**读 A：`As[threadRow*8 + i][dotIdx]`**，As 是 [128][8]，word 地址 = 行 × 8 + 列。取 i = 0、dotIdx = 0：

```
lane 0–15  (threadRow=0) → As[0][0] → word 0  → bank 0
lane 16–31 (threadRow=1) → As[8][0] → word 64 → bank 64 % 32 = 0   ← 同一个 bank
```

1. 32 个 lane 只有 2 种 threadRow。
2. 只有 **2 个不同地址**，各被 16 个 lane broadcast；两者相差 64 个 word，是 32 的倍数，撞进同一个 bank → **2-way**。原因：As 每行只有 8 个 float，往下跨 8 行正好 64 个 word，绕 bank 两圈回到原地。
3. 每个值：本 thread 用 8 次，16 个 lane 共享。

**读 B：`Bs[dotIdx][threadCol*8 + j]`**，Bs 是 [8][128]，word 地址 = dotIdx × 128 + threadCol × 8 + j。取 j = 0、dotIdx = 0：

```
threadCol:  0   1   2   3 | 4   5   6   7 | 8  ...  | 12 ... 15
word:       0   8  16  24 | 32  40  48  56 | 64 ...  | 96 ... 120
bank:       0   8  16  24 | 0   8  16  24  | 0  ...  | 0  ...  24
```

1. threadCol = 0..15，每个出现两次。
2. **16 个不同地址**，相邻相差 8 个 word → 只落在 bank 0、8、16、24 这 4 个 bank，每个 bank 4 个不同地址 → **4-way**（按标量 LDS 算）。原因：每个 thread 拿连续 8 个，相邻 thread 的起点被"撑开"了 8 个 word。
3. 每个值：本 thread 用 8 次，2 个 lane 共享。

### 4.4 为什么 1D 没有冲突、2D 有

| | 读 A | 读 B |
|---|---|---|
| 1D | threadRow 全 warp 相同 → 同一个地址 → broadcast | threadCol 连续 32 个 → 相邻差 1 个 word → 铺满 32 个 bank |
| 2D | 2 种 threadRow → 2 个地址相差 64 word → 2-way | 每人连续 8 个 → 相邻差 8 个 word → 4-way |

> 1D 里每个 thread 只拿 1 列，相邻 lane 的地址要么相同、要么紧挨着。2D 里每个 thread 要拿一块 8×8，相邻 lane 之间就被撑开了：A 撑开 64 个 word，B 撑开 8 个 word，一对上 32 的倍数就撞 bank。

这是 2D 的代价：**让一个 thread 多干活（复用更多），warp 里的地址就变稀疏了。** 下一课的办法：转置 As（`As[8][128]`），用 `float4` 读，让地址重新挤紧。

## 5. 算账

### 5.1 smem：指令数 vs wavefront

| | smem kernel | 1D | 2D |
|---|---:|---:|---:|
| 每 thread 每 dotIdx：smem 读 / FMA | 2 / 1 | 9 / 8 | 16 / 64 |
| 每次 smem 读换几次 FMA | 0.5 | 0.89 | **4** |
| 每 warp 每 dotIdx：FFMA 指令 | 1 | 8 | 64 |
| 每 warp 每 dotIdx：smem wavefront（标量 LDS + 冲突，理论） | 2 | 9 | 8×2 + 8×4 = **48** |
| wavefront / FFMA 指令（越小越好） | 2 | 1.13 | 0.75 |

- 只数指令，2D 比 1D 好 4.5×（4.00 / 0.89）；算上冲突、按 wavefront 只好约 1.5×。实测 1.26×，更接近后者。
- **只数指令会高估收益。**

### 5.2 粗略上限（推测）

- H100 每个 SM 每拍能发 4 条 warp-FFMA（4 个 SMSP），处理约 1 个 smem wavefront。
- 每 warp 每 dotIdx：64 条 FFMA 要 16 拍；48 个 wavefront 要 48 拍 → smem 是瓶颈，计算单元约 2/3 的时间在等。
- 上限 ≈ 16 / 48 × 67T ≈ **22T**；实测 22.21T。
- **但不可靠**：同一个模型套在 1D 上得 2/9 × 67 ≈ 14.9T，比 1D 实测 17.56T 还低。说明编译器至少在 1D 里做了源码以外的事，比如把 As 合并成 LDS.128。2D 的"吻合"可能是巧合。方向（smem 读是瓶颈）比较有把握，数字要靠 SASS / ncu 定。
- 另一种说法：`worklog.md` 认为 2D 的主要瓶颈是 "GMEM instruction count"，同样没测过。两种说法都待验证。

### 5.4 SASS 实测（2026-10-03，推翻了上面的几个结论）

来源：kernel op session，报告 `~/reports/blocktile-sass-2026-10-03.md`。二进制和 job 221866/221870 是同一个（`build_matmul.sh`，nvcc `-O3 --use_fast_math`，sm_90）。

**每个 thread、每轮 K-tile（8 个 dotIdx）**：

| | LDG | STS | BAR | LDS（32 位） | LDS.128 | FFMA |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 8 | 8 | 2 | 0 | **32** | 512 |
| 1D | 2 | 2 | 2 | 8 | 16 | 64 |

- **2D 的 As**：16 条 `LDS.128`，**不是** 64 条 32 位 `LDS`。dotIdx 循环完全展开后，`As[r][0..7]` 这一行在 smem 里是连续的 8 个 float，编译器一条 `LDS.128` 拿 4 个 dotIdx 的值（`As[r][0..3]`），每行 2 条 × 8 行 = 16 条。Simon 在 A6000 上的 SASS 里 As 还是 32 位 `LDS`，我们的 H100 build 不一样，原因没查。
- **2D 的 Bs**：16 条 `LDS.128`（每个 dotIdx 2 条，各拿连续 4 个），和 Simon 的一样。
- **1D 的 As** 也被合并成了 16 条 `LDS.128`。这解释了为什么我的模型套在 1D 上会低估（第 5.2 节）。
- **寄存器 162 个**、无 spill → 每个 SM 只放 1 个 block。

**这推翻了什么**

| 原来的说法 | 实测后 |
|---|---|
| 每 thread 每 dotIdx 16 条 smem 读 | 4 条 `LDS.128`（2 As + 2 Bs） |
| As 读 2-way、Bs 读 4-way（按 32 位） | 不适用。`LDS.128` 的 bank 规则 NVIDIA 没完整公开。kernel op 按"quarter-warp 分组处理"推测：As 是 broadcast，无冲突；Bs 可能 2-way。**未用 ncu 验证** |
| 每 warp 每 dotIdx 48 个 wavefront → smem 是瓶颈 → 上限约 22T | 4 条 `LDS.128`，就算每条要 4 个 wavefront 也只有 16 个，和 64 条 FFMA 的 16 拍持平。**smem 很可能不是主要瓶颈**，"22T 吻合"是巧合 |
| 寄存器 100–128，每 SM 2 个 block | 162 个，每 SM 1 个 block（8 个 warp，每个调度器只有 2 个 warp） |

**新的嫌疑（推测，未测）**：每个 SM 只有 1 个 block，而这个 block 每轮都是"搬运（LDG，几百拍延迟）→ sync → 计算 → sync"串行进行。搬运等数据时，SM 上没有别的 block 能顶上来计算。要验证需要 ncu 看 occupancy 和 stall 原因（`long_scoreboard`、`barrier`）。这正是后面双缓冲（rung 12）要解决的问题。

**教训**：源码字面 ≠ 实际指令，别人的 SASS ≠ 我们的 SASS。分析访存之前先看自己 build 的 SASS。

### 5.3 global / L2 / HBM（跨 block 复用）

**block 内部**：

- 一个 block 一生要读 A 的 128 行 × 4096 = 2MB，加 B 的 4096 × 128 = 2MB，共 4MB。
- 但它**不需要同时持有这 4MB**：沿 K 一片一片走，每片 8KB 进 smem，用完就扔，以后再也不回头。所以 block 内每个元素从 global 只读一次，不靠 cache 装下整条。smem 就是手动管理的 cache。

**block 之间**（重复发生在这里）：

```
block 数     = (4096/128)² = 1024
每个读 4MB   → 向 L2 的总请求量 ≈ 4.3GB
A + B 本身   = 64MB + 64MB = 128MB
→ 每个字节平均被请求 32 次（同一行 32 个 block 读同一条 A，同一列 32 个 block 读同一条 B）
```

**HBM 实际流量（推测）**：

- blockIdx.x 变得最快，所以同一时刻在跑的约 264 个 block（132 SM × 推测 2 个）≈ C 的前 8 行多一点；（2026-10-03 更正：SASS 显示每 SM 只能放 1 个 block，所以同时只有约 132 个 block，约 8 波；B 的 HBM 流量变成约 64MB × 8 ≈ 0.5GB，合计约 0.6GB、约 3% 的时间，结论"HBM 不是瓶颈"不变）
- 它们大致齐步沿 K 走，在任一 K 步合起来只碰 A 的 9 片 + B 的 32 片 ≈ 164KB，L2（50MB）轻松装下；
- 1024 / 264 ≈ 4 波：A 每条只在一波里用（共 64MB），B 每波要整个（64MB × 4）→ HBM ≈ 0.3GB。

| | 流量 | 时间 | 占 kernel 6.2ms |
|---|---:|---:|---:|
| L2 → SM | 4.3GB | ≈ 0.7TB/s | 远低于 L2 带宽 |
| HBM → L2 | ≈ 0.3GB（推测） | 0.3GB ÷ 3.35TB/s ≈ 0.1ms | 约 1.5% |

结论（推测）：HBM 几乎闲着，瓶颈在 SM 内部。L1 在这里几乎不起作用：复用靠 smem（手动）和 L2（跨 SM 共享）。

**block tile 大小决定跨 block 的重复量**：

```
总读请求 = N³ × 4B × (1/BM + 1/BN)
```

| 课 | block tile | 总读请求 |
|---|---|---:|
| 第 3 课 smem | 32×32 | 17.2GB |
| 第 4 课 1D | 64×64 | 8.6GB |
| 第 5 课 2D | 128×128 | 4.3GB |

主动安排 block 执行顺序、让同时跑的 block 多共享 L2，在仓库后面的 `wgmma_v9_3`（GROUP_M rasterization swizzle）里才做：N=8192 时 +25.6T。到那时 Tensor Core 快了 20 倍以上，HBM / L2 才成为瓶颈。

## 6. 实测

| kernel（FP32，N=4096，H100） | TFLOPS | 占 cuBLAS（51.93T） | 对上一级 |
|---|---:|---:|---:|
| 1D blocktile（64,64,8,8） | 17.56 | 33.8% | — |
| **2D blocktile（128,128,8,8,8）** | **22.21** | **42.8%** | **1.26×** |
| 2D 自动调参（128,128,16,16,8） | 33.78 | 65.1% | +52% |
| 下一课 vectorized（128,128,8,8,8） | 32.73 | 63.0% | — |

- 来源：`RESULTS.md`。
- 下一课把 As 转置、用 float4，同样的参数从 22T 到 32.7T，和"2D 的问题在 smem 怎么读"一致（推测）。
- 仓库里**没有**用 ncu 测过 2D 的 bank conflict（第 7 节最后一问）。

## 7. 问答记录

### 2026-10-02 第一轮（凌晨，太密，中断）

第一次讲第 5 课时，我一条回复里塞了整套讲解、bank 冲突、算账和 6 处文档问题。你凌晨 2 点读了 10 分钟没读懂。之后改成**一次只讲一小块，讲完等你说"懂了"再往下**：第 1 块（8×8 + 外积）→ 第 1.5 块（L2 / HBM）→ 第 2 块（搬运循环）→ 第 3 块（bank conflict）。

**Q：这个 kernel 有多复杂、多高级？一般要学多久？世界上有多少人真懂？**

A：

- 2D blocktile 本身是**中级**：经典 matmul 教程（siboehm）的第 5 步，约 100 行，核心概念只有 tiling 和 register blocking。难的不是代码，是分析（warp 视角、bank、wavefront）。
- 在仓库阶梯里大约在 1/3 处。最难、最稀缺的是后面的 WGMMA / TMA / warp specialization。
- 学习时间（粗略估计，无可靠数据）：吃透经典 7 步要几周到 2 个月；独立写出不错的 GEMM 并会用 ncu 找瓶颈要 3–6 个月；Hopper 上接近 cuBLAS 再加 6–12 个月以上；cuBLAS / CUTLASS 团队那种级别要好几年。
- 人数（粗猜）：懂 tiling / register blocking 思想的几万到十万；能做 warp 级访存分析的几千到一万；能在 H100 上用 WGMMA + TMA 写到接近 cuBLAS 的几百到一两千。
- 决定速度的是亲手测了多少（改参数、看 SASS、跑 ncu、猜瓶颈再验证），不是读了多少。

### 2026-10-02 第二轮（分块重讲）

**Q（你的理解）：每个 thread 负责 C 里一整个 8×8 tile？负责 A 的 8 行和 B 的 8 列？每次只读 8 个 A、8 个 B，外积，16 个值变 64 个？**

A：都对 ✅。补充时间上的细节：整个 kernel 跑下来，确实用到 A 的 8 整行和 B 的 8 整列（K 方向全部 4096 个），但每个 dotIdx 只拿一个切片：A 是这 8 行在第 k 列上的 8 个数（As 的一小段竖条），B 是这 8 列在第 k 行上的 8 个数（Bs 的一小段横条）。

**Q（你的假设）：这 64 个值是部分和，也放在寄存器里？**

A：对 ✅。每个 K 步往里累加一次，4096 步全走完才一次性写回 C。

**Q：我们处理的是 FP32？寄存器是什么类型？**

A：这个 kernel 全是 FP32。寄存器本身不分类型，都是 32 位格子，一个 float 正好占 1 个（第 2.2 节）。

**Q（你的假设）：需要 64 个累加寄存器 + 16 个输入寄存器？**

A：对 ✅，源码层面是 80 个；再加地址、下标，推测实际 100–128 个，没跑 `ptxas -v`。

**Q（你的假设）：一个 warp 负责 C 里什么形状？8×256？还是 64×128？**

A：都不对 ❌，是 **16 × 128**。warp 里 tid 0..15 是 threadRow 0、threadCol 0..15，tid 16..31 是 threadRow 1 → 2 排 × 16 个 8×8 方块。8×256 不对是因为 block 只有 128 宽；64×128 不对是因为 32 个 thread × 64 = 2048 个元素，而 64×128 = 8192。一个 block = 8 个 warp 叠起来 = 128×128。

**Q：warp 负责的是 128×16，不是 16×128，对吗？**

A：按"行 × 列"的惯例，一个 warp 是 **16 行 × 128 列**（矮而宽）。如果你说的"128 by 16"是"宽 128、高 16"，那和 16×128 是同一个形状，只是写法不同。之后统一用"行 × 列"。

**Q（你的假设）：一个 block 在一个 SM 上，负责整个 128×128，所以要读 A 的整 128 行和 B 的整 128 列？**

A：对 ✅。补充：一个 block 从头到尾只在一个 SM 上跑，但一个 SM 可以同时放好几个 block（本 kernel 推测 2 个）。

**Q（你的假设）：设计应该也考虑 L1 / L2 的复用、减少 HBM → cache 的搬运。是不是要让这 128 行 A 和 128 列 B 都装进 cache，HBM → cache 才只搬一次？能装进 L1 吗？L2 应该够？**

A：方向对，但机制和你想的不一样：

- "要考虑跨 block 的复用"✅ 对，这一层真正重要。
- "要把整条装进 cache"❌ 不需要。block 沿 K 流式处理，每片 8KB 用完就扔、不再回头，所以 block 内每个元素本来就只从 global 读一次（第 5.3 节）。
- "能装进 L1 吗"❌ 一个 block 的整条就有 4MB，L1 每 SM 只有 256KB。L1 在这里几乎不起作用。
- "L2 应该够"✅ 对，但够的是"同时在跑的 block 在当前 K 步的工作集"（约 164KB），不是全部 A + B（128MB > L2 的 50MB）。
- 算账：向 L2 请求约 4.3GB，每字节平均被请求 32 次；HBM 推测只约 0.3GB，占 kernel 时间约 1.5%。HBM 不是瓶颈。
- 设计里一直在考虑这件事：block tile 越大，跨 block 的重复越少（17.2 → 8.6 → 4.3GB）。主动优化 L2 共享（GROUP_M swizzle）在 `wgmma_v9_3` 才做。

**Q：还需要知道什么？**

A：还剩两块：第 2 块搬运循环（每人搬 4 个，搬运阶段是干净的，第 3.5–3.7 节、第 4.2 节），以及第 3 块计算阶段的 bank conflict（第 4.3 节）。第 3 块是本课最难也最重要的部分，直接引出下一课。

**Q：我们以前跑的 ncu 有没有测出 bank conflict？**

A：**没有。仓库里从没用 ncu 测过 2D 的 bank conflict。**

| 文件 | 测了什么 | 2D bank conflict |
|---|---|---|
| `docs/ncu-profiling-2026-05-30.md` | N=1024，抓了 17 个 kernel，2D 是 launch 4 | ❌ 抓了但表里没列 2D，也没有任何 bank conflict 指标 |
| `matmul/NCU-PROFILING-2026-10-02.md` | 只有 wgmma 系列 | ❌ |
| 第 1 课 ncu（`matmul_naive.notes.md` 第 5 节） | 只有 naive、coalesced | ❌ |

仓库里关于 2D bank conflict 只有两处，都不是 ncu 测量：

- `worklog.md:676-690` 的纸面分析：A 2-way，和本课结论一致，但**没提 B 的 4-way**。
- `autotune.md:521-534` 加 padding 前后的计时：是 vectorized_auto kernel，不是 2D；而且前后在不同节点上跑（共享 vs 独占），数字有混淆，不能当冲突的证据。

**Q：1D 有没有 bank conflict？为什么只有 2D 有？**

A：1D 没有。原因是 warp 形状，也就是相邻 lane 的地址差多少（第 4.4 节）。1D 相邻 lane 差 0（A，broadcast）或 1（B，连续）；2D 差 64（A）和 8（B）。（这是按 32 位读取算的；2026-10-03 SASS 显示 2D 实际是 `LDS.128`，见第 5.4 节。）

### 2026-10-03 第三轮（核实 bank conflict）

**Q：Simon 的博客有没有讲这个 kernel 的 bank conflict？**

A：没有。读了原文（siboehm.com/articles/22/CUDA-MMM）：

- Kernel 5（2D）一节完全没提 bank conflict。他看到的现象是"指令大多是 load、stall 很多"。
- 他在 Kernel 6 转置 As，是为了让 As 也能用 `LDS.128`，不是为了消除冲突。
- Kernel 6 之后他提到"遇到了 smem bank conflict"，专门写了 Kernel 7、8 去消除。冲突是消掉了，但整体反而更慢，所以没写进正文。
- 全文没有 ncu 的 bank conflict 数据。
- 他贴的 Kernel 5 SASS（A6000）：Bs 是 `LDS.128`，As 是 32 位 `LDS`。
- 68.7% 是他的 Kernel 5 在 **A6000、未调参** 时的结果（15.97T）；Kernel 6 是 18.24T（78.4%）。

**Q：你确定有 bank conflict 吗？**

A：当时的回答：As 2-way 理论上比较有把握但没测过；Bs 4-way 不确定（Bs 可能是 `LDS.128`）；冲突是不是主因也不确定。

后来 SASS 证明，连 As 也是 `LDS.128`，"As 2-way" 在我们的 build 里也不成立（第 5.4 节）。我对 As 的"比较有把握"过于自信了，因为它依赖"编译器照源码字面发 32 位读"这个没验证的前提。

**Q：怎么才能确定？**

A：三把工具：

| 工具 | 回答什么 |
|---|---|
| SASS（`cuobjdump -sass`） | 编译器实际发了什么指令，不需要 GPU |
| ncu | 硬件实际用了多少 wavefront、有没有冲突 |
| padding 对照实验（`As[128][9]`） | 冲突要不要紧（消掉后会不会变快） |

做法：先写预测，再测，看哪个假设被推翻。用 1D 当对照组，校准指标。你选了先做 SASS，交给 kernel op session（结果见第 5.4 节）。

## 8. 我说错过 / 纠正过的地方

- **第一轮讲得太密**：一条回复塞了整课，还在凌晨 2 点。之后改成分块讲。这不是知识错误，但影响学习，记下来以后避免。
- **第一个内嵌交互图的标签写错**：写成 `Bs[d][16·tc+j]`，应为 `Bs[d][8·tc+j]`。正式页面 `matmul_2d_blocktile.html` 里是对的。
- **我自己的上限模型不自洽**：按"标量 LDS + bank 冲突"算 2D 得 22T，和实测吻合；但同一个模型算 1D 得 14.9T，比实测 17.56T 还低。所以 2D 的吻合不能当证据，已在第 5.2 节标明。
- **（2026-10-03 SASS 推翻）As 2-way、Bs 4-way、smem 是 2D 主要瓶颈**：这些都建立在"编译器照源码发 32 位 `LDS`"上，没先看 SASS。实际 As、Bs 都是 `LDS.128`（第 5.4 节）。我对 As 2-way 说"比较有把握"，过于自信。
- **（2026-10-03 SASS 推翻）寄存器 100–128、每 SM 2 个 block**：实际 162 个、每 SM 1 个 block。第 5.3 节的 HBM 估算因此要改（结论不变）。
- **68.7% 的出处说错了**：我说是"siboehm 在 A100 上自动调参后的数字"，实际是他的 Kernel 5 在 A6000、未调参时的结果。

### 学习时理解偏、已纠正的地方

- "一个 warp 负责 8×256 或 64×128" → 16 行 × 128 列。
- "要把 block 需要的整条 A、B 装进 cache，HBM 才只搬一次" → 流式处理，用完就扔，不需要装下整条；跨 block 的重复靠 L2。
- "能装进 L1" → 一个 block 的整条 4MB，L1 只有 256KB；L1 在这里几乎没用。

## 9. 仓库文档里的问题（未修改）

- `matmul_2d_blocktile.h:22`："Expected ~68.7% of cuBLAS (1.9x over 1D)" 是 siboehm 的 Kernel 5 在 A6000（未调参）上的数字；本机实测 42.8%、1.26×。
- `worklog.md` 2D 自动调参一节把 68.7% 称作 "Simon's autotuned A100"，GPU 和"调参"都不对（原文是 A6000、未调参的 Kernel 5）。
- `RESULTS.md:82`：说 `As[BM][BK+1]` padding "built INTO 2D/vectorized"。实际 `matmul_2d_blocktile*.cu` 和 `matmul_vectorized.cu` 都没有 padding，只有 `matmul_vectorized_auto.cu:33` 有。
- `worklog.md` Step 5 "Why the load phase…"：说搬 A 时 "32 threads of a warp all read the same row… coalesced 128B"。BK = 8 时一个 warp 读的是 4 行 × 8 个 float，只有 B 是同一行。
- `worklog.md` Step 5："16 SMEM reads → 64 madds" 只数指令、没算冲突；"total reuse factor = 8 × 8 = 64" 混了概念（每个值复用 8 次，64 是 FMA 数）；"4× 1D" 实际是 4.00 / 0.89 ≈ 4.5×。
- `worklog.md` Step 5：说自动调参类 "see matmul_2d_blocktile.cu"，实际在 `matmul_2d_blocktile_auto.cu`。
- `worklog.md` 自动调参 Lesson 2：用 "likely spilling" 解释 TM/TN 不对称，没验证。另一种推测：TN = 16 时一个 warp 跨 4 个 threadRow，As 读变成 4-way 冲突。
- `worklog.md:676-690`：说 2D 的 A 读是 2-way 冲突。这是按 32 位读取推理的；我们 build 的 SASS 里 As 是 `LDS.128`（broadcast），这个结论不适用。还说 2D 的主要瓶颈是 "GMEM instruction count"，没测过。
- `autotune.md:521-534`：padding 前后对比用了不同节点（共享 `-16` vs 独占 `-27`），提升幅度和节点差异混在一起；"winner is register/compute-bound" 也没测过。
- `docs/ncu-profiling-2026-05-30.md`：抓了 2D kernel（launch 4）但表里没有 2D 列；MFU 分母用的是 TF32 峰值 495T，不是 FP32 的约 67T。
- `matmul_2d_blocktile_tuned.cu:128,131`：BK = 24 的配置不满足 `NUM_THREADS % BK == 0`，会写出 smem 边界（已知问题）。

### 待验证

- ✅ SASS：`LDS` / `LDS.128` / `FFMA` 数量、寄存器数（2026-10-03，第 5.4 节）。
- ⬜ ncu：1D、2D 的 smem bank conflict 数（`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`）和 wavefront 数，确认 `LDS.128` 实际有没有冲突；
- ⬜ ncu：occupancy 和 stall 原因（`long_scoreboard`、`barrier`），检验"每 SM 只有 1 个 block，搬运延迟没被盖住"这个新嫌疑；
- ⬜ ncu：`dram__bytes_read.sum`（推测约 0.6GB）和 L2 读流量（推测约 4.3GB）。

## 10. 要点 / 自测题

- 2D = 寄存器这一层两边都复用：16 次 smem 读换 64 次 FFMA，每个值用 8 次。
- 一个 thread 算 8×8；一个 warp 是 C 里 16 行 × 128 列的一条；一个 block 是 8 条叠成 128×128。
- 64 个部分和一直在寄存器里，最后一次性写回。
- block 内部流式处理，每个元素只从 global 读一次；跨 block 的重复（每字节约 32 次）由 L2 吸收，HBM 不是瓶颈（推测）。
- 搬运阶段是干净的。计算阶段按源码字面算会有 bank conflict（A 2-way、B 4-way），但 SASS 显示编译器把读取合并成了 `LDS.128`，每 thread 每 dotIdx 只有 4 条 smem 读。smem 很可能不是主要瓶颈；新嫌疑是寄存器 162 个 → 每 SM 只有 1 个 block（推测）。
- 先看自己 build 的 SASS，再分析访存：源码字面 ≠ 实际指令，别人的 SASS ≠ 我们的 SASS。
- 判断 bank 冲突：同一条指令里，相邻 lane 的地址差几个 word。差 0 或 1 没问题，差 8、64 会撞。
- "读几次"不等于"花几拍"：要数 wavefront，不能只数指令。
- 同一条原则用了第三次：HBM → smem（第 3 课）、smem → 寄存器只复用 B（第 4 课）、A 和 B 都复用（本课）。

<details><summary>自测 1：tid = 17 的 thread 在计算阶段负责 C 块的哪几行、哪几列？</summary>

threadRow = 17 / 16 = 1，threadCol = 17 % 16 = 1 → 行 8..15，列 8..15。
</details>

<details><summary>自测 2：tid = 200 的 thread 在搬运阶段搬哪 4 个 A、哪 4 个 B？</summary>

innerRowA = 25、innerColA = 0 → As 第 25、57、89、121 行的第 0 列。
innerRowB = 1、innerColB = 72 → Bs 第 1、3、5、7 行的第 72 列。
</details>

<details><summary>自测 3：warp 3 在 C 块里负责哪些行？</summary>

warp 3 = tid 96..127 → threadRow 6、7 → C 块第 48..63 行，全部 128 列。
</details>

<details><summary>自测 4：如果 As 转置成 <code>As[8][128]</code>，读法变成 <code>regA[i] = As[dotIdx][threadRow*8 + i]</code>，warp 0 读 i = 0、dotIdx = 0 时几个不同地址？几-way？</summary>

word 地址 = dotIdx × 128 + threadRow × 8 + i：threadRow 0 → word 0（bank 0），threadRow 1 → word 8（bank 8）。2 个不同地址在 2 个不同 bank，各自 broadcast 给 16 个 lane → 无冲突，1 个 wavefront（原来是 2-way）。这就是第 6 课转置 As 的原因。
</details>

<details><summary>自测 5（按 32 位读取的假设）：autotune 冠军 BM = BN = 128、BK = 16、TM = 16、TN = 8（128 thread）。一个 warp 里 threadRow 有几种？A 读几-way？每 dotIdx 的 FFMA / wavefront 是多少，比 (8,8) 好还是差？</summary>

threadCol = tid % (128/8) = tid % 16，threadRow = tid / 16 → warp 里 2 种 threadRow。

As 是 [128][16]，地址 = (16 × threadRow + i) × 16 + d → 两组相差 256 个 word，是 32 的倍数 → 2-way。B 读和现在一样，4-way。

每 dotIdx：FFMA 16 × 8 = 128 条；wavefront = 16 条 A 读 × 2 + 8 条 B 读 × 4 = 64 → 0.5 wavefront / FFMA，比 (8,8) 的 0.75 好。这和它更快一致（推测，没测）。
</details>

<details><summary>自测 6（按 32 位读取的假设）：为什么 1D 没有 bank conflict，2D 有？</summary>

1D 的 warp 里 threadRow 全相同（A 同地址 broadcast）、threadCol 连续（B 相邻差 1 个 word）。2D 每个 thread 拿 8×8，warp 里有 2 种 threadRow（A 两组相差 64 word → 同 bank），每人连续 8 个（B 相邻差 8 word → 只用 4 个 bank）。
</details>

<details><summary>自测 7：整个 kernel 向 L2 请求多少字节？为什么 HBM 实际流量小得多？</summary>

N³ × 4B × (1/BM + 1/BN) = 4096³ × 4 × (2/128) ≈ 4.3GB。同时在跑的约 264 个 block 大致齐步沿 K 走，当前工作集只有约 164KB，每片从 HBM 来一次后被很多 block 从 L2 读走，推测 HBM 只约 0.3GB。
</details>

<details><summary>自测 8：为什么编译器能把 <code>As[threadRow*8 + i][dotIdx]</code> 合并成 <code>LDS.128</code>？源码里一个 dotIdx 内的 8 个 As 明明不连续。</summary>

dotIdx 循环被完全展开了。展开后，同一行 r 在 dotIdx 0..7 上的 `As[r][0..7]` 是 smem 里连续的 8 个 float。编译器把读取顺序重排：一条 `LDS.128` 一次拿 `As[r][0..3]`，供 dotIdx 0–3 使用。每行 2 条 × 8 行 = 16 条。代价是这些值要提前放进寄存器，这也是寄存器用到 162 个的原因之一（推测）。
</details>

