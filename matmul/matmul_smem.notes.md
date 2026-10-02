# 第 3 课学习笔记：shared memory tiling

[可视化页面](matmul_smem.html) · 上一课：[coalesced](matmul_coalesced.notes.md) · 下一课：[1D blocktile](matmul_1d_blocktile.notes.md)

源码：`matmul_smem.cu` / `matmul_smem.h` · H100 · FP32 · N=4096

---

## 1. 核心思路

在 coalesced kernel 里，一个 32×32 的 block 负责 C 的一个 32×32 小方块。

- 这 32 行 C 都要用 A 的同样 32 行，这 32 列 C 都要用 B 的同样 32 列。
- 但每个 thread 都自己跑完整个 k 循环，各发 2N 次 global load。
- 结果：同一个 `A[row][k]` 被同一个 block 里 32 个不同的 thread 各 load 一次。L1/L2 能接住一部分，但每条 load 指令都要走一遍 LSU 和 cache 查找，延迟也高。

smem tiling 的做法：**block 里所有 thread 一起，把 A 和 B 的一个 32×32 tile 从 global memory 搬进 shared memory，每个元素只读一次 global memory，然后 32 个 thread 共用。**沿 K 方向一块一块地走。

## 2. 基础概念

### shared memory（smem）

- SM 上的一块片上 SRAM，和 L1 是同一块物理存储。
- **程序员自己管理**：放什么、什么时候放、放在哪，都由代码决定。cache 则是硬件自动决定的。
- **block 内共享**：thread 0 写进去，thread 500 能读到；不同 block 之间互相看不见。
- **延迟低且固定**：约二三十个 cycle（L2 两三百个 cycle）。
- `__shared__ float As[32][32];` 声明的数组整个 block 只有一份，不是每个 thread 一份。这是第一次出现"变量属于 block 而不是 thread"。

### `__syncthreads()`

- block 内所有 thread 的栅栏：所有 thread 都执行到这一行，才有人能继续往下走。
- 不同 warp 的执行进度没有任何保证，所以凡是"A 写、B 读"跨 thread 的数据，都需要栅栏。

### smem bank

- smem 分成 32 个 bank，每个 bank 宽 4B。连续的 4B 地址依次落在 bank 0、1、2…31、0、1…
- 一个 warp 的 32 个 lane 同时访问 smem 时：
  - 32 个地址落在 32 个不同 bank → 1 次完成，无冲突。
  - 多个 lane 读**同一个地址** → broadcast，不算冲突。
  - 多个 lane 读**同一个 bank 的不同地址** → bank conflict，要分多次完成。
- 一个 SM 每个 cycle 大约能服务 1 次这样的访问（32 bank × 4B = 128B），叫一个 wavefront。

## 3. 逐行代码讲解

```cuda
#define SMEM_TILE 32
__shared__ float As[SMEM_TILE][SMEM_TILE];   // 32×32×4B = 4KB
__shared__ float Bs[SMEM_TILE][SMEM_TILE];   // 4KB，一个 block 共 8KB
```

```cuda
int tx = threadIdx.x;  int ty = threadIdx.y;
int row = blockIdx.y * 32 + ty;
int col = blockIdx.x * 32 + tx;
```

- 映射和 coalesced 一样：`tx` → 列，`ty` → 行。
- block 是 32×32，按 warp 展平规则，**一个 warp = 同一个 ty，tx 取 0..31**。后面所有分析都基于这一点。

```cuda
for (int tileIdx = 0; tileIdx < N; tileIdx += SMEM_TILE) {
```

- k 原来一个一个走，现在每次跨 32。N=4096 时外层共 128 轮，每轮"先搬、再算"。

```cuda
As[ty][tx] = A[row * N + (tileIdx + tx)];   // 越界时填 0
Bs[ty][tx] = B[(tileIdx + ty) * N + col];
```

- 1024 个 thread 各搬 1 个 A、1 个 B，正好填满两个 32×32 tile。
- 两个 load 都是 coalesced：warp 内 tx 连续 → 同一行连续 32 个 float = 128B = 4 个 sector。
- 比 coalesced kernel 更好的一点：那里 A 是 broadcast，一个 warp 只拿到 1 个有用的 float；这里一个 warp 一次拿 32 个不同的 A。
- 越界填 0：N 不是 32 的倍数时，0 不影响点积。

```cuda
__syncthreads();   // 第一个：RAW
```

- thread (ty=3, tx=5) 接下来要读 `As[3][0..31]`（warp 3 的 32 个 thread 写的）和 `Bs[0..31][5]`（32 个不同 warp 写的）。不等大家写完就读，会读到旧值或垃圾。

```cuda
#pragma unroll
for (int k = 0; k < SMEM_TILE; k++)
    sum += As[ty][k] * Bs[k][tx];
```

- 和 naive 的点积公式一样，只是数据来源换成 smem，而且只算这一块的 32 个 k。
- `#pragma unroll` 把循环展开成直线代码，省循环计数和分支。展开后编译器**可能**把 `As[ty][k..k+3]` 合并成一条 `LDS.128`（同一行连续）。**未看 SASS，推测。**

```cuda
__syncthreads();   // 第二个：WAR
}
```

- 下一轮一开始就要覆盖 As/Bs。快的 warp 先进入下一轮写的时候，慢的 warp 可能还在读这一块。
- 漏掉第二个 sync 是很典型的 bug：测的时候碰巧对，换个 N 就错。

```cuda
C[row * N + col] = sum;   // coalesced 写回
```

- launch：`dim3 threads(32, 32)`，grid = (N/32, N/32)。`blockDim` 参数被忽略，`-b` 无效。

## 4. 访存分析：单个 thread 视角 vs 整个 warp 视角

代码 `sum += As[ty][k] * Bs[k][tx]` 写的是**单个 thread**。以 thread (ty=3, tx=5) 为例：

```
k=0:  As[3][0]  * Bs[0][5]
k=1:  As[3][1]  * Bs[1][5]
...
k=31: As[3][31] * Bs[31][5]
```

- A 沿第 3 行从左往右走，B 沿第 5 列从上往下走。

但硬件按**整个 warp** 执行。warp 3 = ty=3，tx=0..31，同一个 k 上：

```
             As[3][k]（A 侧）           Bs[k][tx]（B 侧）
lane 0  →    As[3][k]                   Bs[k][0]
lane 1  →    As[3][k]                   Bs[k][1]
...
lane 31 →    As[3][k]                   Bs[k][31]
             32 个 lane 同一地址           第 k 行整行，32 个 bank 各一个
```

| 读 | 32 个 lane 的地址 | 结果 |
|---|---|---|
| `As[ty][k]` | 全部相同 | broadcast，1 次访问 |
| `Bs[k][tx]` | 连续 32 个 float | 无 bank conflict，1 次访问 |

- 32 个 thread 各自"往下走的一列"并排放在一起，每个 k 上看到的就是 Bs 的一整行。

## 5. 算账

| 每个 thread（N=4096） | coalesced | smem tiling |
|---|---|---|
| global load | 2N = 8192 | 2·N/32 = 256 |
| smem load | 0 | 8192 |
| `__syncthreads` | 0 | 256 |

global load 少了 32 倍，但速度只提高 1.59 倍。**瓶颈换成了 smem**：

- 每个 warp 每个 k：2 次 smem 访问（A broadcast + B 一行）→ 32 次 FMA。
- 一个 SM 每 cycle 约 1 次 smem 访问 → 每 cycle 最多 16 次 FMA，而 FP32 单元每 cycle 能做 128 次 → 只用到约 1/8。
- 67T × 1/8 ≈ 8.4T。若 A 被合并成 `LDS.128`，上限约 13T。实测 9.0T 落在区间内。
- **这是推算，未用 SASS / ncu 验证。**

## 6. 实测

| kernel（FP32，N=4096，H100） | TFLOPS | 占 cuBLAS（51.93T） |
|---|---|---|
| coalesced | 5.70 | 11.0% |
| smem tiling | 8.99 | 17.3% |

来源：`RESULTS.md`；coalesced 对照来自 job 221866。

## 7. 问答记录

**Q：动画里改 k 时，B 换到另一整行，A 只挪一格，A 看起来是 broadcast。代码真的是 `As[ty][k] * Bs[k][tx]` 吗？**

A：是的，你看到的是对的。图画的是整个 warp，代码写的是单个 thread。单个 thread：A 沿行右移、B 沿列下移。整个 warp：A 是 broadcast（只亮一格），B 是第 k 行整行（32 个 lane 各一个）。把 32 条竖线并排，就是"一行一行往下换"。

**Q（感想）：GPU 计算真的很难懂。**

A：难在同一行代码要同时从两个角度读：代码按单个 thread 写，性能由 32 个 thread 合在一起的行为决定。到目前为止的分析都可以归结为**三个问题**：

1. 这个 warp 的 32 个 lane，`tx`、`ty`（或 `threadIdx.x`）各是多少？
2. 代进下标，32 个地址长什么样？全部相同 = broadcast；连续 = coalesced / 无 bank conflict；跳着隔开 = strided，坏。
3. 这个数据会被用几次？用一次就丢是浪费。

**Q：理解 GPU kernel 还有什么建议？**

A：

1. 先预测，再测量。预测落空才是最有价值的时候（例：发现旧 naive 本来就是 coalesced）。
2. 用 warp 视角看访存（上面三个问题）。
3. 记住几个硬件数字，学会算账：FP32 ~67T、BF16 tensor ~989T（dense）、HBM ~3.35 TB/s、smem 每 SM 每 cycle 128B、每 SM 最多 2048 thread / 228KB smem。
4. 读 SASS（`cuobjdump -sass`），数 `LDG` / `LDS` / `FFMA` / `.128`。
5. 把 ncu 当裁判，每次只验证一个假设。
6. 故意搞坏它：删一个 sync、改 tile 大小、转置存储，先预测再测。
7. 合上代码自己从头写一遍。
8. 对所有文档保持怀疑，包括自己的和 Claude 的。
9. 资源：siboehm 的 matmul 博客、《Programming Massively Parallel Processors》第 4–6 章、GPU MODE 讲座；讲到 Hopper 再看 Colfax / CUTLASS。

建议的每课流程：预测 → 讲解 → 验证（SASS / ncu）→ 动手改 → 用自己的话复述。

## 8. 我说错过 / 纠正过的地方

- 讲这一课时我说 A 的 broadcast "几乎是白花的"（32 个 lane 只拿到 1 个有用的 float）。**这是夸大**：broadcast 的那个值 32 个 lane 都用了，同样换来 32 次 FMA。真正的问题是每个从 smem 读出的值在每个 lane 里只用 1 次（下一课纠正）。
- 那时我说导航栏已包含在 commit `7924927` 里，实际上没有；后来在 `86efd88` 补交。

## 9. 仓库文档里的问题（未修改）

- `worklog.md:134` 的 bank 分析：说 `As[ty][k]` "ty 在 thread 间变化、步长 32 → 不同 bank"。实际上一个 warp 的 ty 相同，是 broadcast；而且若 ty 真的变化，步长 32 个 float 会落在**同一个** bank，是 32-way conflict。结论（无冲突）碰巧对，推理错。
- `MatmulSmem` 接收 `blockDim` 但不用，`-b` 无效。
- `matmul_smem.h` 的 "Expected ~12.8% of cuBLAS" 是 siboehm 在 A6000 上的数字，不是本机实测。

## 10. 要点 / 自测题

- smem 是 block 共享、程序员管理的片上存储；tiling 让每个 global 元素只读一次、被 32 个 thread 共用。
- 两个 `__syncthreads` 分别防 RAW 和 WAR，缺一不可。
- global load 少 32 倍但只快 1.59 倍：瓶颈转到 smem，每次 FMA 仍要读 2 次 smem。

<details><summary>自测 1：删掉第二个 <code>__syncthreads()</code> 会怎样？</summary>

快的 warp 进入下一轮，覆盖 As/Bs，慢的 warp 还在读上一块 → 读到错误数据。结果时对时错，取决于 warp 调度，很难复现。
</details>

<details><summary>自测 2：把 <code>SMEM_TILE</code> 从 32 改成 64 会怎样？</summary>

64×64 = 4096 个 thread，超过一个 block 最多 1024 个 thread 的上限，"每个 thread 搬一个、算一个"的写法直接不成立。要么每个 thread 搬多个、算多个——这正是下一课 1D blocktile 的方向。smem 用量 2×64×64×4B = 32KB，本身放得下。如果能做到，每个 thread 的 global load 会再减半（N/64 轮）。
</details>

<details><summary>自测 3：如果把映射换成 <code>tx = threadIdx.y; ty = threadIdx.x</code>（于是 warp 内 ty 变化、tx 固定），<code>As[ty][k]</code> 会怎样？</summary>

32 个 lane 访问 `As[0..31][k]`，地址间隔 32 个 float，全部落在同一个 bank → 32-way bank conflict，要 32 次才能完成。这正是 worklog.md:134 推理错误的地方。
</details>
