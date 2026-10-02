# 第 2 课：Coalesced kernel 学习笔记

[可视化页面](matmul_coalesced.html)

上一课：[matmul_naive.notes.md](matmul_naive.notes.md) · 下一课：[matmul_smem.notes.md](matmul_smem.notes.md)

> 这份笔记整理自 2026-10-01 的教学对话：讲解、问答、ABAB 实测（job 221866）、SASS 对比和 ncu 报告。
> 这一课的大部分讨论发生在 commit `9b2fb4d` **之前**，那时仓库里的 “naive” 还是旧版（`threadIdx.x → col`），本来就是 coalesced 的。所以本课很多内容其实是在比较“旧 naive 16×16 / 旧 naive 32×32 / coalesced”。
> 和现在真正 uncoalesced 的 naive 的对比放在第 1.3 节和第 5.5 节。

---

## 1. 核心思路

### 1.1 这个 kernel 做了什么

- 还是“一个 thread 负责 C 的一个元素”，循环体和 naive 一样。
- 区别在于 thread 编号怎样换算成 (row, col)：用 **1D 的 1024 个 thread**，手动拆成 `threadRow = threadIdx.x / 32`、`threadCol = threadIdx.x % 32`。
- `BLOCKSIZE = 32` **写死**，忽略 harness 的 `-b` 参数。
- 结果是一个 warp（连续 32 个 `threadIdx.x`）= C 上的 **1 行 × 32 列**：
  - 读 B 是连续 32 个 float = 128B = 4 个 sector，coalesced。
  - 读 A 时 32 个 lane 读同一个地址，是 broadcast。
  - 写 C 是连续的，coalesced。

### 1.2 这一课最重要的发现（历史）

- 旧版 naive 本来就是 `threadIdx.x → col`，已经 coalesced 了。
- coalesced 的拉平方式和 2D 32×32 block **完全一样**，所以它的**访存模式等价于 `naive -b 32`**。
- 那这一级到底改了什么？只改了两件事：
  1. 把 block 固定成 32×32（旧 naive 默认是 16×16）。
  2. 换了一种 C++ 写法（`A_row[k] * B_col[k*N]`），编译出的指令顺序略有不同。
- 实测：旧 naive 32×32 比 16×16 快 **15.9%**；coalesced 比旧 naive 32×32 **慢 6.8%**（job 221866，CV < 0.07%）。

### 1.3 现在的 ladder（`9b2fb4d` 之后）

naive 已经改成真正 uncoalesced（`threadIdx.x → row`），所以 coalesced 这一级现在名副其实：

- coalesced 5.699T 对新 naive 16×16 0.965T（**5.90×**）、对新 naive 32×32 0.498T（**11.44×**），都是 job 221870 同一次测的。

---

## 2. 基础概念

### 2.1 Warp 是怎么由 2D thread 拉平出来的（复习，这一课的核心）

```
linear_id = threadIdx.x + threadIdx.y * blockDim.x
warp      = linear_id / 32
```

| block | warp 0 包含的 thread | 对应 C 的形状 |
|---|---|---|
| 旧 naive 16×16 | (x=0..15, y=0) + (x=0..15, y=1) | 2 行 × 16 列 |
| 旧 naive 32×32 | (x=0..31, y=0) | 1 行 × 32 列 |
| coalesced（1D 1024） | threadIdx.x = 0..31 → row 0，col 0..31 | 1 行 × 32 列 |

“同一个 warp 的 row 都一样”这个直觉，**只在 blockDim.x ≥ 32 时成立**。

### 2.2 Sector 粒度与地址合并

- 内存访问的最小单位是 **32B sector**（8 个 float）。一条 128B cache line = 4 个 sector。
- 从 Volta 开始，L1 向 L2 按 sector 取数据，不会强制取满 128B。
- **同一个 warp 内重复的地址只取一次**。这一点决定了 16×16 时 B 并没有“读两遍”（见第 4 节）。
- broadcast（同一地址）占 1 个 sector；coalesced（32 个连续 float）占 4 个；strided 每个 lane 各占 1 个。

### 2.3 Block 层面的数据重用（L1）

一个 warp 一次 load 碰几个 sector 只是一方面。同一个 block（同一个 SM）里的其他 warp 也会读同样的 A 行、B 列。这些数据如果还留在 L1 里，就不用再去 L2 取。block 越大，同一份数据被越多 thread 共用，L2→L1 的流量越少。

### 2.4 Instruction issue + memory latency bound

旧 naive / coalesced 每轮都是 2 条 load 加 1 条 FMA，总指令数和 block 形状无关。硬件的大部分时间花在发 load、等数据回来上。所以改 block 形状只能带来十几个百分点的提升，不会是数量级的变化。

### 2.5 SASS 和几种指令

`cuobjdump -sass` 能看到 GPU 实际执行的机器码。CUDA C++ 只表达意图，真正跑的是 SASS。

- `LDG`：global load。
- `FFMA`：一次 FP32 乘加。
- `IMAD.WIDE`：64 位地址计算。
- 注意要用 mangled name 才能单独 dump 一个 kernel，例如 `_Z21matmulCoalescedKernelPKfS0_Pfi`。

### 2.6 ncu 里本课用到的几个 stall / 指标

- **long_scoreboard**：在等 global load 的数据返回。
- **lg_throttle**：LSU 队列满，load 发不出去。
- **issue active**：SM 有多少 cycle 在发指令。
- **L1 sector hit rate**、**L2 sectors 读**：衡量 cache 重用。
- ncu 默认会把 GPU 时钟锁在 base clock，所以它测出来的时间比例可能和不锁频的 bench 不一样。

### 2.7 为什么很多教材默认 16×16

- **cc 1.x（2006–2009，Tesla 架构）**：每个 block 最多 512 个 thread，32×32 = 1024 根本不允许。而且那时 coalescing 按 **half-warp（16 个 thread）**进行，16 宽正好对齐硬件。
- **Fermi（cc 2.0，2010）之后**：上限提高到 1024，coalescing 也改成按整个 warp。
- 但 Programming Guide 和许多教科书的例子一直用 16×16，这个习惯就流传下来了。
- **block 大小永远由程序员在 `<<<blocks, threads>>>` 里指定**，不存在“编译器默认 block size”。本仓库的 16 是 harness 写死的默认值（`matmul.cpp:1136`）。worklog 给的理由是 “Safest default — 256 threads works on any GPU”。

---

## 3. 逐行代码讲解

文件：`matmul_coalesced.cu`

```cuda
__global__ void matmulCoalescedKernel(const float *A, const float *B, float *C, int N) {
    const int BLOCKSIZE = 32;
```

block 边长写死成 32，不读 `-b`。

```cuda
    int threadCol = threadIdx.x % BLOCKSIZE;
    int threadRow = threadIdx.x / BLOCKSIZE;
```

block 是 1D 的（`threadIdx.x` = 0..1023）。用 `%` 和 `/` 手动拆成二维坐标。连续 32 个 `threadIdx.x` 的 `threadRow` 相同、`threadCol` = 0..31，这正是“一个 warp = 一行 × 32 列”的来源。

```cuda
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = blockRow * BLOCKSIZE + threadRow;
    int col = blockCol * BLOCKSIZE + threadCol;
```

和旧 naive 一样：`blockIdx.x` 对应列方向，`blockIdx.y` 对应行方向。grid 都是 128×128。

```cuda
    if (row < N && col < N) {
        float sum = 0.0f;
        const float *A_row = A + row * N;   // 指向 A 第 row 行的开头
        const float *B_col = B + col;       // 指向 B 第 col 列的顶端
        for (int k = 0; k < N; k++) {
            sum += A_row[k] * B_col[k * N];
        }
        C[row * N + col] = sum;
    }
}
```

- 先算好行指针和列指针，循环里再用偏移。数学上和 `A[row*N+k] * B[k*N+col]` 完全一样。
- 但这种写法让编译器排出了**不同的 load 顺序**（见 5.3 节），这是 6.8% 差距最可能的来源（**推测**）。
- 第 45 行的注释说 A “threads in same warp access different rows (not coalesced for A)”，这是**错的**：同一个 warp 的 row 相同，读 A 是 broadcast（见第 8 节）。

### Host 端

```cuda
void MatmulCoalesced::execute(const float *d_A, const float *d_B, float *d_C) {
    const int BLOCKSIZE = 32;
    dim3 threads(BLOCKSIZE * BLOCKSIZE);              // 1D，1024 个 thread
    dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE,
                (N + BLOCKSIZE - 1) / BLOCKSIZE);
    matmulCoalescedKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
}
```

构造函数收下了 `blockDim`，但从来不用。

### 和旧 naive 的代码对比

```cuda
// 旧 naive：2D block，大小由 -b 决定（默认 16）
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
sum += A[row * N + k] * B[k * N + col];

// coalesced：1D block，32 写死
int row = blockIdx.y * 32 + threadIdx.x / 32;
int col = blockIdx.x * 32 + threadIdx.x % 32;
sum += A_row[k] * B_col[k * N];
```

---

## 4. 访存分析：单个 thread 视角 vs 整个 warp 视角

### 4.1 单个 thread 视角

每个 thread 都一样：A 沿第 row 行往右走（`A[row*N + k]`），B 沿第 col 列往下走（`B[k*N + col]`）。这个视角下旧 naive、coalesced 没有任何区别。

### 4.2 整个 warp、一步 k 的 sector 数

| 一个 warp 的一次循环 | 旧 naive 16×16 | 旧 naive 32×32 / coalesced |
|---|---|---|
| 读 A | 2 个不同地址（两行），**2 sectors**，相隔 16KB | 1 个地址（broadcast），**1 sector** |
| 读 B | 两个半 warp 读**同一组** 16 个连续 float = 64B = **2 sectors** | 32 个连续 float = 128B = **4 sectors** |
| 合计 / 32 次 FMA | **4 sectors** | **5 sectors** |

关键：B 的地址 `B[k*N + col]` **只和 k、col 有关，和 row 无关**。16×16 时两个半 warp 的 col 都是 c..c+15，读的是同一组 64B，硬件把它们合并，只取一次。所以 16×16 并没有“读两倍数据”；按单个 warp 算，它取的 sector 反而更少（4 对 5）。

两种情况下 A 都很浪费：每取一个 sector（8 个 float），这一步只用 1 个。不过接下来 7 步的 k 正好会用到同一个 sector里的另外 7 个，前提是它还留在 L1 里。

ncu 实测的 load sectors/request：16×16 是 **2.0**（4 sectors / 2 条 load），32×32 和 coalesced 都是 **2.5**（5 / 2），和上表一致。

### 4.3 Block 层面：每 8 轮 k 从 L2 取多少（当时是推测）

一个 A sector 含 8 个 float，读进 L1 后接下来 7 轮都能命中。以 8 轮 k 为单位：

| | 16×16（256 线程） | 32×32（1024 线程） |
|---|---:|---:|
| A：每行 1 个 sector | 16 | 32 |
| B：一行 k 上的列 | 2 | 4 |
| 合计 sector | 18 | 36 |
| 换来的 FMA | 256 × 8 = 2048 | 1024 × 8 = 8192 |
| **每 1000 次 FMA 需要的 sector** | **8.8** | **4.4** |

block 越大，同一份数据被越多 thread 共用，L2→L1 流量减半。后来的 ncu 证实了 **2 倍这个比例**，但绝对值估小了：实测是 15.5 对 7.7（见 5.4 节）。

### 4.4 和真正 uncoalesced 的新 naive 对比（32×32）

| | 新 naive（x→row） | coalesced（x→col） |
|---|---:|---:|
| A sectors | 32 | 1 |
| B sectors | 1 | 4 |
| 合计 / 32 次 FMA | 33 | 5 |
| 取回数据的利用率 | ≈ 12.5% | ≈ 82.5% |

---

## 5. 实测与 profiling

### 5.1 测量协议（job 221866）

- gcp5 Slurm，`general` partition，`--exclusive`，1 张 H100，时限 30 分钟。脚本是 gcp5 上的 `~/naive-blk.sbatch`，log 是 `~/naive-blk_221866.log`。
- 你要求“多跑几次看分布和平均值，降低噪声再比较”，所以用 **10 轮 ABAB 交替**：奇数轮顺序是 b16 → b32 → coal，偶数轮反过来。
- 每个样本 = `matmul_bench` 一次运行 = 10 次 warmup + 100 次 batched 迭代的平均（CUDA events），N=4096。
- 脚本里的 ncu 步骤失败了（`ERR_NVGPUCTRPERM`，Slurm 节点不允许读 performance counter）；`nvidia-smi` 不在 PATH 里，没记到时钟。

### 5.2 Bench 结果（job 221866，gcp5-h100-0-38，用时 2 分 11 秒）

| 配置 | 平均 TFLOPS | 中位数 | sd | CV | 最小 ~ 最大 |
|---|---:|---:|---:|---:|---:|
| 旧 naive 16×16 | **5.281** | 5.279 | 0.0033 | 0.063% | 5.279 ~ 5.290 |
| 旧 naive 32×32 | **6.118** | 6.118 | 0.0003 | 0.006% | 6.117 ~ 6.119 |
| coalesced | **5.699** | 5.699 | 0.0004 | 0.007% | 5.699 ~ 5.700 |

- 噪声极低（CV < 0.07%），所以下面的差距都是真实的。
- 32×32 比 16×16 快 **15.9%**，和 worklog 记录的 5.3 对 6.1 一致。
- coalesced 比旧 naive 32×32 慢 **6.8%**，而且稳定可复现。

### 5.3 SASS 对比（gcp5 登录节点上用 `cuobjdump` dump）

编译器把 k 循环展开了 16 次。主循环：

| | 旧 naive（`-b 32`） | coalesced |
|---|---:|---:|
| 主循环指令数 | 73 | **70** |
| `LDG` | 32 | 32 |
| `FFMA` | 16 | 16 |
| 地址计算（`IMAD.WIDE` 等） | 约 25 | 约 22 |
| 第一个 `FFMA` 之前已发出的 `LDG` | **≈ 22**（第一个 FFMA 在 `0x03f0`） | **≈ 16**（在 `0x03c0`） |

- 指令数不能解释 6.8%：coalesced 反而少 3 条。
- 开头的 `SHF / LOP3 / LEA` 算出来的 row、col 和 naive 一致。两者寄存器用量相同（ncu：32 个/thread），occupancy 也相同。
- **推测**：差别在 load 的排列。naive 在第一次用到数据之前有更多 load 在路上，等待时间能更好地重叠；coalesced 更早卡住等数据。

### 5.4 ncu（pod `kdev-profiling`，GPU 0；报告 `~/reports/naive-blk-ncu-2026-10-01.md` 第 1–4 节）

> 这几节结论写在操作 session 的报告里，教学对话中没有逐条讨论过。

| 指标 | 旧 b16 | 旧 b32 | coalesced |
|---|---:|---:|---:|
| ncu kernel 时间 (ms) | 30.95 | **26.58** | 27.11 |
| 执行的指令数 | 9.82e9 | 9.82e9 | **9.42e9** |
| global load 指令 | 4.295e9 | 4.295e9 | 4.295e9 |
| L1 sectors / request | 2.0 | 2.5 | 2.5 |
| L1 sector hit rate | 87.5% | 95.0% | 95.2% |
| L2 sectors 读 | 1.062e9 | 0.531e9 | 0.509e9 |
| DRAM 读 | 16.9 GB | 8.45 GB | 4.26 GB |
| achieved occupancy | 99.5% | 99.6% | 99.8% |
| issue active | 37.9% | **44.3%** | 41.9% |
| L1 throughput（峰值占比） | **99.4%** | 77.5% | 76.4% |
| stall: long_scoreboard | **24.16** | 9.81 | 16.15 |
| stall: lg_throttle | 9.22 | 10.99 | 9.58 |

**假设 1：b16 和 b32 的 load 指令数相同，但 b32 读的 L2 sector 约为一半 → 证实。**

- load 指令数都是 4.295e9。
- L2 sectors 正好 2.00 倍。
- 比例对了，但每 1000 次 FMA 的绝对值实测是 15.5 对 7.7，我估的 8.8 对 4.4 偏小约 1.76 倍。
- 额外发现：旧 b16 的 L1 wavefront 管线打到 99.3%，b16 真正卡在 L1 本身（报告里标为推断）。

**假设 2：coalesced 慢是因为 load 调度 → 基本证实，但幅度没有完全解释。**

- coalesced 指令少 4.1%，long_scoreboard 却高 65%（16.15 对 9.81），warp latency/inst 也高 6%（38.1 对 36.0）。这符合“load 发得不够早”的说法。
- 但 ncu 下只慢 **2.0%**，bench 慢 **6.8%**。ncu 锁频可能压缩了差距。
- 这是相关性，不是因果。最直接的验证是把两者的 load 顺序对调后再测，**还没做**。

**未解释**：coalesced 的 DRAM 读只有旧 b32 的一半（4.26 对 8.45 GB），L2 hit 62% 对 41%，可是两者的 grid 映射一样。DRAM 不是瓶颈，所以影响不大。

### 5.5 和新 naive 对比（job 221870，gcp5-h100-0-35，同样的协议）

| kernel | mean GFLOPS | sd | coalesced 是它的几倍 |
|---|---:|---:|---:|
| 新 naive 16×16 | 965.27 | 0.17 | 5.90× |
| 新 naive 32×32 | 498.15 | 0.01 | 11.44× |
| coalesced（对照） | 5698.78 | 0.55 | 1× |

coalesced 在两次 job 里都是 5.699T，所以换节点没有影响。新 naive 的分析见[第 1 课笔记](matmul_naive.notes.md)第 5 节。

---

## 6. 问答记录

**Q1. “是不是因为 16×16 和 32×32 的 block 问题？默认值是多少？”**（在你提出“naive 已经是 coalesced 的”这个判断之后）

答：对，旧 naive 和 coalesced 唯一真正的区别就是 block 大小。默认 `blockDim = 16`（`matmul.cpp:1136`），coalesced 写死 32。16×16 时一个 warp 跨两行，所以“同一个 warp 的 row 相同”只在 blockDim.x ≥ 32 时成立。

**Q2. 你的假设：“16×16 时 32 个 thread 跨两行，每行读 16×4 = 64 bytes。但 HBM 一次读 128 bytes，所以第一行读 128、第二行读 128，即使 coalesced 也读了两倍数据，所以慢一点；32×32 一次 128 bytes 就够整个 warp 用。”**

答：**一半对，一半错**。

- ✅ 16×16 时一个 warp 跨两行：对 C 是对的；对 A 也算对（读 A 的两个不同 row，但每个只有 4 bytes）。
- ❌ “B 读两行 × 64 bytes”：B 的地址和 row 无关，两个半 warp 读的是**同一组** 64 bytes，硬件会合并。
- ❌ “HBM 每次读 128 bytes”：基本单位是 32B sector，L1 按 sector 去 L2 取；而且这个 kernel 大部分读取都在 L1/L2 命中，没有到 HBM。
- 按 sector 计数，16×16 每个 warp 每步 4 个 sector，32×32 是 5 个，前者反而更少。

**Q3. 你的猜测：“默认 16×16 是因为 FP64 时那是最优 block size；FP32 要加倍。也许是 Ampere 时代的编译器默认 16×16，Hopper 时代会默认 32×32？只是随便猜。”**

答：**不对**。

- 不存在编译器默认 block size，block 大小永远由程序员在 `<<<>>>` 里指定。
- 和 FP64 无关：warp 从第一代 CUDA GPU 起就是 32 个 thread，和数据类型无关。
- 但“有历史原因”这个直觉方向是对的：cc 1.x 每个 block 最多 512 个 thread，而且按 half-warp（16 个）合并访存。16×16 是那个时代留下的惯例，被教材一直沿用。

**Q4. “读数据这部分我还是不太懂，能画图解释吗？”接着：“能不能把整个 A 和 B 都画出来？A 可以设成 40×40，不会太大。”**

答：先画了一张 SVG sector 图（一个 warp、一步 k，16×16 对比 32×32）。之后画了可交互的完整 40×40 图：A 在左下、B 在右上、C 在右下；可以切 block 大小、拖 warp 和 k、显示整个 k 循环。40 不能被 16 或 32 整除，所以边缘的 block 有一部分 thread 被 `if (row < N && col < N)` 挡掉。这些都已写进 `matmul_coalesced.html`。

**Q5. “这样看来，16×16 和 32×32 不会有明显的性能差距？”**

答：**基本正确**。单看 warp 的 coalescing，两者没有本质区别，而且瓶颈（每轮 2 条 load + 1 条 FMA 的发射和延迟）完全一样。但实测确实差约 15%。当时**推测**这来自 block 层面的 L1 重用（8.8 对 4.4 sectors / 千次 FMA）。后来 ncu 证实了 L2 流量正好减半（4.3 节、5.4 节）。

**Q6. “你知道怎么跑测试吗？能再跑一次吗？”以及“可以多跑几次，拿到速度的分布和平均值，对吧？目的是降低噪声、读数准确再比较。”**

答：对，这正是 ABAB 协议的目的。提交了 job 221866（10 轮交替，偶数轮反序）。结果的 CV < 0.07%，所以 15.9% 和 6.8% 这两个差距都可以放心当成真实的。ncu 在 Slurm 上没有权限。

**Q7. “可以在 H100 集群上开 dev pod 跑 ncu（比如 gcp5 k8s），可能已经有 pod 了。”**

答：在 namespace `ziliang-xp` 找到了已在运行的 `kdev-profiling` pod（只能用 GPU 0），采集了 b16/b32/coal 三份 ncu 报告。之后你要求这个对话专注学习，操作交给新开的 session “Profile naive vs coalesced matmul with ncu”，由它解析 ncu 并写报告。

**Q8. “之前画的图很好，能写成 HTML 文件吗？以后每个 kernel 一个。”** 以及放置位置的两次反馈：

答：先放在 `~/reports/matmul-viz/`，你说不要放 reports，要放 kernels 仓库；然后放进 `matmul/viz/`（commit `c164e60`），你说要放在和 kernel 同一个位置；最终是 `matmul/matmul_coalesced.html`（`7924927`），前后导航在 `86efd88` 加上。页面开头注明：页内的 “naive” 指 `9b2fb4d` 之前的旧版。

**Q9. “我们的 coalesced kernel 用的是 32×32 block 吗？”**

答：是。`BLOCKSIZE = 32` 写死，1D 的 1024 个 thread，忽略 `-b`。

---

## 7. 我说错过/纠正过的地方

1. **“naive 是 32×32 block”**：第一课照 RESULTS.md 的标签讲的。实际默认是 `-b 16`，5.29T 是 16×16 的结果（详见第 1 课笔记第 7 节）。
2. **“coalesced 在数学上完全等价于 `naive -b 32`；6.1T 和 5.73T 来自不同 session，可以看作测量误差”**：访存模式确实相同，但**性能不等价**。同一节点 ABAB 实测 coalesced 慢 **6.8%**，CV < 0.07%，不是误差。
3. **“差别只可能来自编译器生成的指令数不同”**：第一次猜的是指令数。SASS 显示 coalesced 反而**少 3 条**，所以改成“load 排列顺序不同”（**推测**，ncu 的 long_scoreboard 部分支持，幅度没完全解释）。
4. **siboehm 那句写乱的话**：我写过“同一个 warp 的 32 个线程读的是 32 个不同 row 的 B...”。正确的是：siboehm kernel 1 里 **A 是 strided，B 是 broadcast**。下一条回复已经更正。
5. **block 层面的 sector 估算**：比例 2× 对了，绝对值（8.8 / 4.4 每千次 FMA）比实测（15.5 / 7.7）小约 1.76 倍。
6. **第一课说 coalesced 只从 5.29 提升到 5.73（约 +8%）**：这两个数来自不同 session，而且 5.29 是 16×16。同一次 job 里的准确说法是：coalesced 比旧 b16 快约 7.9%，比旧 b32 慢 6.8%。

---

## 8. 仓库文档里的问题（未修改）

以下只是记录，这份笔记没有改动这些文件。

- **`matmul_coalesced.cu:45`**（对话里我说的是第 47 行，现在的文件在第 45 行）：注释 “A[row][k] - threads in same warp access different rows (not coalesced for A)” 是错的。同一个 warp 的 row 相同，读 A 是 broadcast。（`matmul_coalesced.h:12` 写的 “Row is broadcast across threads” 反而是对的。）
- **`matmul_coalesced.cu:9` 起的 “In naive kernel:” 注释**：描述的是旧 naive 的映射，`9b2fb4d` 之后已经过时。
- **`matmul_coalesced.h:17`**：“Expected ~8.5% of cuBLAS (6.4x over naive)” 是 siboehm 在 A6000 上的数字，不是本机实测。（本机现在是 5.90×，16×16 默认配置。）
- **`worklog.md` Step 1 的 “fatal flaw” 推理有争议**：它说 16×16 时 “B load: two transactions reading the same 16 addresses twice (redundant work)”。同一个 warp 内重复的地址会被合并，16×16 每步取的 sector 反而更少。32×32 更快的真正原因是 block 层面的 L1 重用（ncu：L2 sectors 正好减半）。
- **`worklog.md` 第 43 行**说 32×32 “hits the 1024-thread block limit, hurting occupancy”。ncu 实测 b32 的 achieved occupancy 是 99.6%，并没有受影响。（这一条是整理笔记时对照 ncu 报告发现的，对话里没有讨论过。）
- **`9b2fb4d` 之后过时的文档**（操作 session 列的清单，没有改）：
  - `matmul/worklog.md:82`（Naive 16×16 5.3T）
  - `matmul/coalesced-16bit-2026-09-26.md:35`（“Coalescing gains nothing on the naive rung”是拿旧的、本来就 coalesced 的 naive 比的）
  - `matmul/blog-comparison-2026-05-30.md:28`
  - `docs/tutorial/07-coalesced-gemm.html:39-70`、`docs/tutorial/06-naive-gemm.html`、`docs/tutorial/gemm-roadmap.html:134`
  - `docs/kb/gemm-variant-taxonomy.md:25`、`docs/kb/sgemm-ladder-h100.md`、`docs/ncu-profiling-2026-05-30.md:65`
  - `scripts/gen_gemmladder_html.py:31, :119` 和生成的 `docs/gemm-ladder/kernels/{naive,coalesced}.html`
  - RESULTS.md 的 naive 一行**已在 `8d39f4f` 更新**；coalesced 一行（5.73）没变。

---

## 9. 要点 / 自测题

**要点**

- 判断是否 coalesced，要看同一个 warp 的 32 个 lane 读的地址；warp 由**拉平后的编号**决定，所以 block 形状很关键。
- 同一个 warp 内重复的地址只取一次；内存粒度是 32B sector，不是 128B。
- 单个 warp 的 sector 数看不出 16×16 和 32×32 的差别，要看 **block 层面的 L1 重用**。
- **访存模式相同 ≠ 性能相同**：指令调度也能造成约 7% 的差距。
- **先降低噪声再比较**：10 轮 ABAB，CV < 0.07%，小差距才可信。
- 真正的问题是数据重用太低。下一课 smem tiling 会主动把数据搬进 shared memory，让整个 block 共用。

<details>
<summary>自测题（点开看答案）</summary>

1. **coalesced kernel 里，一个 warp 读 A 和读 B 分别是什么模式？各占几个 sector？**
   答：A 是 broadcast，1 个 sector；B 是 32 个连续 float，4 个 sector。

2. **旧 naive 16×16 时，两个半 warp 读 B 会不会读两遍？为什么？**
   答：不会。`B[k*N + col]` 和 row 无关，两个半 warp 的 col 相同，读同一组 64B，硬件在 warp 内合并。

3. **旧 naive 32×32 比 16×16 快 15.9%，load 指令数相同。差距从哪来？**
   答：block 更大，L1 里的数据被更多 thread 共用：L1 命中率 95% 对 87.5%，L2 sectors 正好减半。

4. **coalesced 和旧 naive 32×32 访存完全一样，为什么慢 6.8%？**
   答：SASS 指令数差不多（coalesced 还少 3 条），但第一个 FFMA 之前在路上的 load 少（约 16 对 22），long_scoreboard 高 65%。这是推测，ncu 下只差 2%，幅度没完全解释。

5. **16×16 这个默认值是怎么来的？**
   答：早期 cc 1.x 每个 block 最多 512 个 thread，而且按 half-warp 合并访存，教材沿用至今。不是编译器默认值，也和 FP64 无关。

6. **为什么 ncu 要在 pod 上跑，而不是 Slurm 节点？**
   答：Slurm 节点没有读 GPU performance counter 的权限（`ERR_NVGPUCTRPERM`）。

</details>
