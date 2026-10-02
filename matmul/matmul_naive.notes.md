# 第 1 课：Naive kernel 学习笔记

[可视化页面](matmul_naive.html)

上一课：无（这是第一级） · 下一课：[matmul_coalesced.notes.md](matmul_coalesced.notes.md)

> 这份笔记整理自 2026-10-01 的教学对话：讲解、问答、实测和 profiling 都在里面。
> 注意有**两个版本的 naive**：
>
> - **旧版**（commit `9b2fb4d` 之前）：`threadIdx.x → col`。讨论后发现它本来就是 coalesced 的。
> - **新版**（`9b2fb4d` 之后，也就是现在仓库里的代码）：`threadIdx.x → row`，和 siboehm kernel 1 一样，是真正 uncoalesced 的。
>
> 本笔记以新版为主线。旧版只在讲历史和对比时出现。旧版和 coalesced 的详细对比（16×16 对 32×32、ABAB、SASS）放在第 2 课笔记里。

---

## 1. 核心思路

- **一个 thread 负责 C 的一个元素**：`C[row][col] = Σ_k A[row][k] · B[k][col]`。这是最直接的分工方式。
- 每轮 k 循环做 2 次 global load（一个 A、一个 B）和 1 次 FMA，算术强度只有 2 FLOP / 8 B = **0.25 FLOP/byte**。
- 线程之间完全不合作，同一份数据被反复读很多次。A 的每一行要被这一行对应的 4096 个 thread 各读一遍，B 的每一列也一样。
- 新版故意把 `threadIdx.x` 映射到 **row**。一个 warp 的 32 个 lane 于是落在 32 个不同的行上：读 A 是 strided，读 B 是 broadcast，写 C 也是 strided。
- 实测（N=4096，FP32）：**16×16 block 0.965 TFLOPS，32×32 block 0.498 TFLOPS**。coalesced 是它的 5.90× / 11.44×。
- 瓶颈是 **L1 的数据通路（wavefront 吞吐）**，不是 HBM 带宽。

---

## 2. 基础概念

### 2.1 CPU 和 GPU 的分工

CPU 只有少数几个核心，每个都很强，擅长复杂逻辑。GPU 有成千上万个简单线程同时跑同一段代码，擅长大量重复计算。矩阵乘法很适合 GPU：N=4096 时 C 有约 1600 万个元素，每个元素的算法完全一样，而且彼此独立。

### 2.2 Kernel

用 `__global__` 标记的函数就是 kernel。它由 CPU（host）发起，在 GPU（device）上执行，而且**同一个函数会被成千上万个 thread 同时执行**。每个 thread 跑的代码一样，区别只在于它们知道自己是第几号，于是各自处理不同的数据。

### 2.3 Thread / Block / Grid

```
Grid（一次 kernel 启动）
 ├── Block (0,0)  Block (1,0)  Block (2,0) ...
 ├── Block (0,1)  ...
 └── ...
每个 Block 里：
   Thread (0,0) (1,0) ... (15,0)
   Thread (0,1) ...
   ...              (15,15)   ← 默认 16×16 = 256 个 thread
```

每个 thread 读到的内置变量不同：

- `threadIdx.x / .y`：我在 block 里的坐标。
- `blockIdx.x / .y`：我所在的 block 在 grid 里的坐标。
- `blockDim.x / .y`：每个 block 有多大。naive 用 harness 的 `-b` 参数，**默认 16**（`matmul.cpp:1136`：`int blockDim = 16;`）。

### 2.4 Warp：32 个 thread 一组，由“拉平”决定

硬件以 **32 个 thread 为一组**执行，这一组叫一个 warp。同一个 warp 的 32 个 lane 在同一时刻执行同一条指令。

硬件不认识 2D。它先把 block 里的 thread 拉平成一维编号，再每 32 个一组：

```
linear_id = threadIdx.x + threadIdx.y * blockDim.x
warp 编号 = linear_id / 32
```

- **32×32 block**：warp 0 = (x=0..31, y=0)，一个 warp 里 `threadIdx.y` 相同，`threadIdx.x` 取 0..31。
- **16×16 block**：warp 0 = (x=0..15, y=0) + (x=0..15, y=1)。一个 warp 跨了两个 `threadIdx.y`。

**这是理解后面所有访存分析的钥匙**：一行代码的性能，取决于“同一个 warp 的 32 个 lane 把下标代进去以后，地址长什么样”。

### 2.5 Row-major 存储

N×N 矩阵在显存（global memory / HBM）里是一条一维数组，按行排列：

```
A[row][k]  存在  A[row * N + k]
```

同一行里相邻的元素，在内存里也相邻。同一列上下相邻的元素，相隔 N 个 float（N=4096 时是 16KB）。

### 2.6 Sector、cache line 和 coalescing

- 内存访问的最小单位是 **32-byte sector**，也就是 8 个 float。
- 一条 **128-byte cache line** = 4 个 sector。
- 从 Volta 开始，L1 向 L2 按 **sector** 取数据。只需要 64 bytes 就只取 2 个 sector，不会强制取满 128 bytes。
- 一个 warp 发出一条 load 指令时，硬件把 32 个地址按 sector 合并：
  - **coalesced（合并访问）**：32 个 lane 读连续的 32 个 float = 128B = 4 个 sector，全部用上。这是最理想的情况。
  - **broadcast**：32 个 lane 读**同一个地址**。硬件只取一次，再广播给所有 lane。只占 1 个 sector，也很好。
  - **strided**：lane 之间的地址相隔很远，每个 lane 各占一个 sector，每个 sector 里只用到 4 字节。这是最差的情况。
- 同一个 warp 内**重复的地址只取一次**。

### 2.7 L1 / L2 / HBM 与数据重用

数据从 HBM 经过 L2、L1 到寄存器。naive 没有主动利用数据重用，但 cache 会被动帮忙：

- 旧版 naive 实测 5.29T，约 26 ms。要读的 2×4096³ 个 float 约 550 GB；如果全部从 HBM 读（约 3.35 TB/s），需要约 160 ms。实际只用了 26 ms，说明**大部分数据是 L1/L2 供应的**。
- 新版 naive 也一样：一个 A sector 装着同一行连续 8 个 k。同一个 lane 接下来 7 步都会在 L1 命中，所以 L1 命中率高达 97–99%。

### 2.8 Wavefront 与 sectors/request（ncu 引入的概念）

- **sectors/request**：一条 warp 级 load 指令平均碰了几个 sector。理想值是 4（32 个连续 float）。这是判断 coalescing 好坏最直接的指标。
- **wavefront**：L1 数据通路的处理单位。**推测**的模型是：一条请求碰到几条不同的 cache line，L1 就要处理几个 wavefront。ncu 里直接统计 wavefront 个数的指标返回了 NA，这个模型是从多组时间比例反推出来的（见第 5 节）。
- **lg_throttle**：LSU 的队列满了，新的 load 指令发不出去。
- **long_scoreboard**：在等 global load 的数据返回。

---

## 3. 逐行代码讲解

文件：`matmul_naive.cu`（`9b2fb4d` 之后的版本）

```cuda
__global__ void matmulNaiveKernel(const float *A, const float *B, float *C, int N) {
```

`__global__` 表示这是 kernel。A、B、C 都是指向 GPU 显存的指针，调用之前矩阵已经放到 GPU 上了。

```cuda
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // x → row
    int col = blockIdx.y * blockDim.y + threadIdx.y;   // y → col
```

每个 thread 用这两行算出自己负责 C 的哪个格子，这是整个 kernel 的核心。

- 新版把 **x 维映射到 row**，这正是 siboehm kernel 1 的写法。
- 按 2.4 节的拉平规则，同一个 warp 的 lane `threadIdx.x` 连续，所以它们的 **row 连续，col 相同**（blockDim.x ≥ 32 时）。
- 举例（16×16 block）：block (2,1) 里的 thread (x=5, y=7)，row = 2×16+5 = 37，col = 1×16+7 = 23，负责 `C[37][23]`。

```cuda
    if (row < N && col < N) {
```

边界检查。N 不是 blockDim 的倍数时，最后一排 block 会多出一些 thread，它们的坐标超出矩阵，什么都不做。

```cuda
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
```

矩阵乘法的定义：A 的第 row 行和 B 的第 col 列做点积。

- `A[row * N + k]`：沿着 A 的第 row 行往右走。
- `B[k * N + col]`：沿着 B 的第 col 列往下走，每步跳 N 个元素。
- `sum` 是局部变量，放在寄存器里。
- 每轮 2 次 global load、1 次 FMA（fused multiply-add）。

```cuda
        C[row * N + col] = sum;
```

写回显存。新版下同一个 warp 的 lane 写的是 32 个不同的行，所以写也是 strided 的。

### Host 端：怎么启动 kernel

```cuda
void MatmulNaive::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(blockDim, blockDim);           // 默认 16×16 = 256 threads
    dim3 blocks((N + blockDim - 1) / blockDim,  // 向上取整
                (N + blockDim - 1) / blockDim);
    matmulNaiveKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
}
```

- `<<<blocks, threads>>>` 是 CUDA 启动 kernel 的语法。
- `(N + b - 1) / b` 是常用的向上取整写法。
- N=4096、默认 `-b 16` 时，grid 是 256×256 = 65536 个 block，每个 block 256 个 thread，一共约 1677 万个 thread，每个 C 元素一个。（第一课里我按 32×32 说成了 128×128 个 block、每个 1024 个 thread。thread 总数一样，但默认配置是 16×16，见第 7 节。）

### 旧版和新版的唯一区别

```cuda
// 旧版（9b2fb4d 之前）：x → col，本来就是 coalesced 的
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// 新版（现在）：x → row，真正 uncoalesced
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;
```

循环体一个字没变，只是 x/y 的分工对调了。性能相差 5.5~12 倍（新 b16 0.965T 对旧 b16 5.281T，新 b32 0.498T 对旧 b32 6.118T），全部来自“一个 warp 的 32 个地址长什么样”。

---

## 4. 访存分析：单个 thread 视角 vs 整个 warp 视角

### 4.1 单个 thread 视角（代码字面意思）

每个 thread 都一样：k 从 0 到 N−1，A 沿自己那一行往右走，B 沿自己那一列往下走。只看这个视角，新旧两版完全没有区别。**性能由下面的 warp 视角决定。**

### 4.2 整个 warp 视角：新版 naive，32×32 block

一个 warp = **32 行 × 1 列**（C 上是一整列）。同一个 k：

| | 32 个 lane 的地址 | sector 数 | 用到的字节 |
|---|---|---:|---:|
| 读 A `A[row*N+k]` | 32 个不同的 row，相隔 N×4 = 16KB | **32**（1024B） | 128B |
| 读 B `B[k*N+col]` | col 相同，同一个地址：broadcast | **1**（32B） | 4B |
| 合计 / 32 次 FMA | | **33** | 132B / 1056B ≈ **12.5%** |

对比 coalesced（x→col）：A 1 个 sector、B 4 个，合计 **5** 个，利用率 132B / 160B ≈ 82.5%。按 sector 数算，naive 每步多取约 **6.6 倍**。

写 C：32 个 lane 写 32 个不同的行，每条 store 碰 32 个 sector（ncu 实测 store 32 sectors/request）。

### 4.3 整个 warp 视角：新版 naive，16×16 block

一个 warp = (x=0..15, y=0) + (x=0..15, y=1) → **16 行 × 2 列**。

| | sector 数 |
|---|---:|
| 读 A：16 个不同的 row（两列共用同样 16 行） | **16** |
| 读 B：2 个相邻的 col，落在同一个 sector | **1** |
| 合计 | **17** |

所以新版下 **16×16 反而比 32×32 快**：一个 warp 每步碰的 cache line 是 17 条对 33 条，33/17 = 1.94，实测 1.93。旧版正好相反（旧版 32×32 更快，见第 2 课笔记）。

### 4.4 L1 能救回多少？

A 的一个 sector 装着同一行连续 8 个 k。拖动可视化页面的 k 滑块，从 0 拖到 7，可以看到 A 的读取点一直在同一批灰色 sector 里往右移：

- 如果这些 sector 还留在 L1，接下来 7 步就不用再去 L2 取。
- 实测 L1 命中率：b32 **99.2%**，b16 **97.0%**，比旧版（95.0% / 87.5%）还高。
- 所以 **L2 和 DRAM 流量和旧版完全一样**（同样的 block 大小相比）。从 HBM 搬上来的字节并没有变多。

既然字节没变多，代价在哪？在 **L1 要处理的次数**：一条 load 指令碰 33 条 cache line，L1 就要一条一条地处理（推测的 wavefront 模型）。L1 数据通路被打满到约 98%，LSU 队列满了，load 发不出去（`lg_throttle` 是头号 stall）。

### 4.5 一句话

**在 H100 上，uncoalesced 的代价不一定是浪费 DRAM 带宽。这一次数据被 L1 吸收了，代价变成 L1 的处理次数暴涨。**

---

## 5. 实测与 profiling

### 5.1 测量方法（两次 job 都用同样的协议）

- 平台：gcp5 Slurm 集群，`general` partition，`--exclusive` 独占节点，1 张 H100。
- `matmul_bench --method naive -n 4096 -b {16|32}`；对照组是 `--method coalesced -n 4096`。
- 每次运行 = 10 次 warmup + 100 次 batched 迭代，用 CUDA events 计时取平均（`matmul.cpp` 约 1022–1048 行）。FLOPs 按 `2N³ − N²` 计算。
- **10 轮 ABAB 交替**，偶数轮顺序反过来，用来抵消时间漂移。每种配置得到 10 个样本。
- ncu 在 Slurm 节点上报 `ERR_NVGPUCTRPERM`（没有读 performance counter 的权限），所以改到 k8s 的 `kdev-profiling` pod（namespace `ziliang-xp`）上跑，只能用 GPU 0。
- Slurm 节点上 `nvidia-smi` 不在 PATH 里（在 `/usr/local/nvidia/bin`），所以没有记录时钟和温度。

### 5.2 旧版 naive（x→col），job 221866，gcp5-h100-0-38

| 配置 | 平均 TFLOPS | sd | CV |
|---|---:|---:|---:|
| 旧 naive 16×16 | 5.281 | 0.0033 | 0.063% |
| 旧 naive 32×32 | 6.118 | 0.0003 | 0.006% |
| coalesced | 5.699 | 0.0004 | 0.007% |

详细分析见第 2 课笔记。

### 5.3 新版 naive（x→row），job 221870，gcp5-h100-0-35

先用 `--verify` 在 N=1024 检查：`-b 16` 和 `-b 32` 都 PASS，max rel err ≈ 2.3e-6。

| 配置 | mean GFLOPS | median | sd | min ~ max | coalesced 是它的几倍 |
|---|---:|---:|---:|---:|---:|
| 新 naive 16×16 | **965.27** | 965.31 | 0.17 | 964.84 ~ 965.40 | **5.90×** |
| 新 naive 32×32 | **498.15** | 498.15 | 0.01 | 498.12 ~ 498.15 | **11.44×** |
| coalesced（对照） | 5698.78 | 5698.58 | 0.55 | 5698.44 ~ 5700.26 | 1× |

- 对照组 coalesced 和 job 221866 的 5.699T 完全一致，所以换节点没有影响。
- siboehm 在 A6000 上测到的差距约 6~8 倍。
- RESULTS.md 的 naive 一行已经改成新数字（commit `8d39f4f`）：0.97T（16×16，默认），32×32 0.50T，约为 cuBLAS FP32 的 1.9%（旧版是 10.2%）。

### 5.4 ncu（pod GPU 0，报告 `~/reports/naive-blk-ncu-2026-10-01.md` 第 5 节）

| 指标 | 新 b16 | 新 b32 | 旧 b32 |
|---|---:|---:|---:|
| ncu 时间 (ms) | 177.8 | 343.6 | 26.6 |
| 指令数 | 9.82e9 | 9.82e9 | 9.82e9 |
| global load 指令 | 4.295e9 | 4.295e9 | 4.295e9 |
| **load sectors / request** | **8.5** | **16.5** | 2.5 |
| store sectors / request | 16 | 32 | — |
| L1 sector hit rate | 97.0% | 99.2% | 95.0% |
| L2 sectors 读 | 1.090e9 | 0.539e9 | 0.531e9 |
| DRAM 读 | 16.8 GB | 8.45 GB | 8.45 GB |
| L1 data pipe（lsu wavefronts）峰值占比 | **97.8%** | **98.3%** | 77.0% |
| L1 lsuin requests 峰值占比 | 11.5% | 6.0% | 77.0% |
| L2 / DRAM throughput | 5.3% / 2.8% | 1.4% / 0.7% | 11.2% / 9.6% |
| issue active | 6.6% | 3.4% | 44.3% |
| warp latency / inst (cycles) | 241 | 460 | 36 |
| stall: lg_throttle | **186.8** | **434.0** | 11.0 |
| stall: long_scoreboard | 46.0 | 17.0 | 9.8 |

怎么读这张表：

- **sectors/request 和预测一致**：b32 一条 A load 碰 32 个、B 碰 1 个，平均 (32+1)/2 = 16.5；b16 是 (16+1)/2 = 8.5。
- **L2、DRAM 流量和旧版一样**：L1 命中率反而更高，HBM 不是瓶颈。
- **L1 数据通路满载（约 98%），但请求数只用到 6–11%**：请求不多，但每个请求要碰几十条 cache line。
- **stall 的形态变了**：主要不是在等数据返回（long_scoreboard），而是 load 根本排不进 LSU（lg_throttle）。
- 用“每条不同的 cache line 一个 wavefront”这个模型（**推测**，wavefront 计数指标返回 NA）检验时间比例：
  - 新 b32 对新 b16：33/17 = 1.94×，实测 1.93×。
  - 新 b32 对旧 b32：wavefront 多 16.5 倍，按 L1 利用率修正（77% → 98%）后约 13×，实测 12.9×。
  - 两组比例都对得上，但模型本身没有被直接测到。

---

## 6. 问答记录

**Q1. “我没怎么写过 CUDA，能从基础开始，带我逐行看 naive kernel 吗？”**

答：先讲 thread/block/grid、内置变量、warp、row-major，再逐行讲代码，最后算账说明为什么慢。核心结论是**数据重用太低**：每次 FMA 读 8 字节，只能靠 cache 被动帮忙。后面所有优化都在回答同一个问题：怎样让读进来的每个字节被更多计算用到。（第一课讲的是旧版代码。）

**Q2. 你的假设：“A 和 B 都是 row-major。同一个 warp 的 thread row 相同，读的是同一个 A 值；col 不同，读 B 是 32 个相邻的 float。所以 naive 其实已经是 coalesced 的？写 kernel 的模型可能写错了。”**

答：**对（针对旧版代码）**。旧版 naive 是 `threadIdx.x → col`，一开始就是 coalesced 的：读 A 是 broadcast，读 B 是连续 128B。所以“coalesced”这一级要修的问题本来就不存在。`matmul_naive.h` 原来的注释说 B 是 “strided reads, BAD”，那是错的（`9b2fb4d` 已改正）。

补充：“同一个 warp 的 row 相同”只在 blockDim.x ≥ 32 时成立。默认 16×16 时一个 warp 跨两行。

**Q3. “是不是 16×16 和 32×32 block 的问题？默认值是多少？”**

答：默认 `blockDim = 16`（`matmul.cpp:1136`，可以用 `-b` 改），所以 naive 默认是 16×16。coalesced 把 32 写死了。旧 naive 和 coalesced 的真正区别就是 block 大小（加上 codegen 的细微差异）。这部分详见[第 2 课笔记](matmul_coalesced.notes.md)。

**Q4. 你对 siboehm naive 的理解：“他故意把 x 分给 row，所以 B 总是读同一个值（好），A 总是 strided，每个 float 相隔 N 个位置（N=4096）。如果正确地把 x 分给 col、y 分给 row，就会是很好的 coalesced 访问。”**

答：**完全正确**。siboehm kernel 1 是 `x = blockIdx.x*blockDim.x + threadIdx.x` 当 row 用，读 `A[x*K + i]` 时 32 个 lane 相隔 K×4 字节（N=4096 时 16KB），完全 strided；读 `B[i*N + y]` 时 y 相同，是 broadcast。（同一条消息里关于 16×16“读了两倍数据”和默认 block size 的部分，放在第 2 课笔记的问答里。）

**Q5. “coalesced kernel 是不是用 32×32 block？另外，能不能把 naive 改成真正的 naive？我想测它的速度。让另一个 profiler agent 去测，再告诉我们结果。再做一个 naive 读内存方式的可视化。”**

答：

- coalesced 写死了 `BLOCKSIZE = 32`，1D 的 1024 个 thread，忽略 `-b`。
- 改法问了你两个选择，你选了推荐项：**直接改 `matmul_naive.cu`**（不新增 method），并且**只改 FP32**（`matmul_naive_typed.cu` 不动）。
- commit `9b2fb4d`：映射改成 `threadIdx.x → row`，`.h` 的注释也一起改正。
- 测速交给操作 session 跑：job 221870 加 ncu，结果见第 5 节。
- 可视化先放在 `viz/01-naive.html`（`df0a894`），后来按你的要求搬到 `matmul_naive.html`（`7924927`），再加上前后导航（`86efd88`）。

**Q6. （我问你）“32×32 时 naive 每步取 33 个 sector，coalesced 取 5 个，单看这个是 6.6 倍。你觉得实测会慢多少？L1 能挽回多少？”**

这道题你没有正式作答（回复是“Yeah, I got it”）。我当时的预期是“L1 能留住 A 的 sector，所以可能慢不到 6.6 倍”。实测是 **32×32 慢 11.44 倍，16×16 慢 5.90 倍**：L1 确实留住了数据（命中率 99.2%），但代价从“字节数”换成了“L1 处理次数”，所以 32×32 反而比 6.6 倍还慢。见第 7 节。

**Q7. “能在 session 里做一个像 coalesced 那样可以交互的可视化吗？”以及“写到 kernel 旁边的正确位置，在浏览器打开，并且让页面互相链接（只要链接上一级和下一级）。”**

答：在对话里放了可交互的 40×40 图，可以在 naive（x→row）和 coalesced（x→col）之间切换，也可以切 16×16/32×32、拖 warp 和 k、显示整个 k 循环。之后写进 `matmul_naive.html`，放在 `matmul_naive.cu` 旁边，带上一级/下一级导航。

---

## 7. 我说错过/纠正过的地方

1. **“naive 用的是 32×32 block”**：第一课照 RESULTS.md 的 “32×32 tile” 标签讲的。查证后发现默认是 `-b 16`（`matmul.cpp:1136`），5.29T 是 16×16 的结果（worklog：16×16 是 5.3T，32×32 是 6.1T）。所以第一课第三节“一个 warp 的 row 相同”的分析，只对 32×32 成立，对默认配置不成立。第一课的 grid 举例（128×128 个 block、每个 1024 个 thread）也是按 32×32 讲的；默认其实是 256×256 个 block、每个 256 个 thread。
2. **siboehm 那段话写乱了**：讲 siboehm 时我写过一句“同一个 warp 的 32 个线程读的是 32 个**不同 row** 的 B...”，这句是错的。正确的是：siboehm kernel 1 里 **A 是 strided、B 是 broadcast**。
3. **“实际可能慢不到 6.6 倍”这个预期错了**：实测 32×32 慢 11.44 倍。原因是代价不在取回的字节数（L1 把它吸收了），而在 L1 要处理的 cache line 条数。
4. **给操作 session 的 “约 17 sectors/request” 估算**：b32 时 A 是 32 个 sector（32 行），不是 16，正确值是 (32+1)/2 = 16.5。17 这个数恰好接近，但算法错了。
5. **（非技术）导航栏的 commit**：我说导航栏已经包含在 `7924927` 里，其实那个 commit 只做了文件搬家，导航改动后来在 `86efd88` 才提交。

---

## 8. 仓库文档里的问题（未修改）

以下只是记录，这份笔记没有改动这些文件。

- `matmul_naive.h` 的旧注释说 “A coalesced, B strided reads – BAD”，这是错的：旧版 B 是 coalesced，A 是 broadcast。**已在 `9b2fb4d` 改正**。
- RESULTS.md 原来把 naive 的 5.29T 标为 “32×32 tile”，但它是 16×16 的结果。**已在 `8d39f4f` 更新**为新版 naive 的数字，并加了脚注。
- `matmul_naive_typed.cu`（FP16/BF16）仍然是旧的 x→col 映射，所以 FP16/BF16 naive 和 FP32 naive 已经不能直接比较。RESULTS.md 用 ʰ 脚注标出来了。
- 下面这些在 `9b2fb4d` 之后已经过时，还在引用旧 naive 的 5.3T / 4.7T，或在旧 naive 的基础上下结论（操作 session 列出来的，没有改）：
  - `matmul/worklog.md` Step 1（第 12–46 行）和第 82 行
  - `matmul/matmul_coalesced.cu:9` 起的 “In naive kernel:” 注释，以及 `matmul/matmul_coalesced.h:17`
  - `matmul/blog-comparison-2026-05-30.md:28`
  - `matmul/coalesced-16bit-2026-09-26.md:35`
  - `docs/tutorial/gemm-roadmap.html:134`、`docs/tutorial/06-naive-gemm.html`、`docs/tutorial/07-coalesced-gemm.html:39-70`
  - `docs/kb/gemm-variant-taxonomy.md:25`、`docs/kb/sgemm-ladder-h100.md`、`docs/ncu-profiling-2026-05-30.md:65`
  - `scripts/gen_gemmladder_html.py:31` 和 `:119`，以及它生成的 `docs/gemm-ladder/kernels/{naive,coalesced}.html`
- 小问题：`matmul_naive.html` 第 2 节的操作提示说“实际速度不一定慢 8 倍”，第 3 节说的是 6.6 倍。8 倍指的是每个 A sector 只用到 1/8，6.6 倍是 33/5 的 sector 比，两个说法口径不同。

---

## 9. 要点 / 自测题

**要点**

- 性能由 **warp 视角**决定：把 32 个 lane 的 `threadIdx` 代进下标，看地址是相同（broadcast）、连续（coalesced），还是分散（strided）。
- warp 由拉平后的编号 `x + y*blockDim.x` 决定，所以 block 的形状会改变一个 warp 在 C 上的形状。
- 新版 naive 慢，不是因为 HBM 不够用，而是 L1 每条请求要处理太多 cache line：sectors/request 16.5，L1 数据通路约 98% 满载。
- 判断 coalescing 好坏，最直接的指标是 **sectors/request**，理想值是 4。

<details>
<summary>自测题（点开看答案）</summary>

1. **新版 naive、32×32 block，warp 0 在 C 上是什么形状？**
   答：32 行 × 1 列（一整列），因为 `threadIdx.x → row`。

2. **同样是新版，换成 16×16 block，一个 warp 每步 k 碰几个 sector？**
   答：A 16 个（16 行），B 1 个（2 个相邻的 col 在同一个 sector），一共 17 个。

3. **为什么新版 16×16 比 32×32 快约 1.94 倍，而旧版正好相反？**
   答：新版的代价和一个 warp 碰到的不同 cache line 数成正比：33 对 17。旧版两种配置都是 coalesced 的，差别来自 block 层面的 L1 重用（详见第 2 课）。

4. **新版 naive 的 DRAM 流量比旧版多吗？**
   答：不多。同样的 block 大小下完全一样，因为一个 A sector 装着 8 个连续的 k，同一个 lane 接下来 7 步都会在 L1 命中（命中率 99.2%）。

5. **头号 stall 是哪个，说明什么？**
   答：`lg_throttle`，说明 LSU 队列满了，load 指令发不出去。瓶颈在 L1 的处理能力，而不是在等 DRAM 返回数据。

6. **默认 block 大小是多少，在哪里定义？**
   答：16，定义在 `matmul.cpp:1136`，可以用 `-b` 修改。

</details>
