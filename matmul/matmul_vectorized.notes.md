# 第 6 课学习笔记：vectorized（float4 读写 + As 转置）

[可视化页面](matmul_vectorized.html)（还没做） · 上一课：[2D blocktile](matmul_2d_blocktile.notes.md) · 下一课：[warptile](matmul_warptile.notes.md) · 总进度：[PROGRESS.md](PROGRESS.md)

源码：`matmul_vectorized.cu` / `matmul_vectorized.h` · H100 · FP32 · N=4096

状态（2026-10-03）：**进行中**。讲完第 1 块（float4 读 A）和第一轮问答。还没看 vectorized 的 SASS，下面凡是标"推测"的都没验证。

---

## 1. 核心思路

### 1.1 第 5 课停在哪

- 2D 外积：每 thread 16 次 smem 读换 64 次 FFMA。
- 我们 build 的 SASS：As、Bs 都已被编译器合并成 `LDS.128`（各 16 条/tile）；162 个寄存器 → 每 SM 1 个 block、8 个 warp（12.5%）。
- ncu：smem stall ≈0；Bs 有 2 倍 bank conflict，不是瓶颈。
- 没回答的问题：同样参数 (128,128,8,8,8)，vectorized 22.21T → 32.73T（+47%），从哪来？

### 1.2 相对第 5 课改了什么

| | 2D | vectorized | 变了吗 |
|---|---|---|---|
| 参数、256 thread、外积、两个 `__syncthreads` | — | — | 一样 |
| ① global 读 A、B | 每 thread 各 4 个 `float`（循环） | 各 **1 个 `float4`**（第 63、83 行） | 改了 |
| ② smem 里 A 的布局 | `As[128][8]` | **`As[8][128]`，转置存**（第 21、73–76 行） | 改了 |
| ③ 计算时读 A | `As[threadRow*8+i][dotIdx]` | `As[dotIdx][threadRow*8+i]` | 跟着 ② |
| ④ 写回 C | 64 次标量写，每个判断边界 | **`float4` 写**（第 147 行） | 改了 |
| ⑤ 边界处理 | 每元素判断，越界填 0 | 每个 `float4` 判断一次 | 改了 |

## 2. 基础概念

### 2.1 向量化访存：`LDG.128` / `LDS.128` / `STS.128` / `STG.128`

- 同一个想法：**一条指令搬 16B（4 个 float）**，代替 4 条 32 位指令。
- 区别只是内存空间：`LDG`/`STG` = global 读/写，`LDS`/`STS` = shared 读/写。
- 条件：4 个 float **连续**，起点 **16B 对齐**。
- 两种来源：源码里写 `float4`（显式告诉编译器）；或者编译器自己证明连续 + 对齐后自动合并（第 5 课 As/Bs 的 `LDS.128` 就是这样来的）。
- 字节数不变，变的是指令数。

## 3. 逐行代码讲解

### 3.1 用 float4 读 A（第 47–48、63 行）

```cuda
const int innerRowA = threadIdx.x / 2;   // 0..127
const int innerColA = threadIdx.x % 2;   // 0..1
tmp = *reinterpret_cast<const float4*>(&A[innerRowA * N + innerColA * 4]);
```

- 单 thread：一条指令读连续 4 个 float（16B）。tid = 5 → 第 2 行第 4–7 列。256 × 4 = 1024 = 整块 128×8，不需要循环。

（其余各行待讲。）

## 4. 单 thread vs warp 访存

### 4.1 搬 A（global）

1. 下标：lane 2r、2r+1 读第 r 行；warp 0 = 第 0–15 行。
2. 地址：每行 8 个 float = 32B = 1 个满 sector；一条指令 16 个 sector，全用满。
3. 每个值搬一次。

对比 2D：一条 32 位 LDG 碰 4 个 sector，4 条才搬完 warp 的 128 个数（第 0–3、32–35、64–67、96–99 行）。**字节、sector 一样，指令 4 → 1。** B 同理（warp 读一行连续 512B）。

另外，2D 里每个 thread 的 4 个 A 在 4 个不同的行（innerRowA + 0/32/64/96），本来就不连续，编译器无法合并成 `LDG.128`。

## 5. 算账

### 5.1 LDG 指令占比（推测）

| 每 thread 每 tile | 2D（SASS 实测） | vectorized |
|---|---:|---:|
| LDG | 8 | 2（推测） |
| FFMA | 512 | 512 |
| 全部（不算地址计算） | ≥ 562 | ≈ 553 |

LDG 只占约 1.4%。即使 issue 是瓶颈，省 6 条也只值约 1%。**"float4 省了 LSU 指令"解释不了 +47%**（推测）。

## 6. 实测

| kernel（FP32，N=4096） | TFLOPS | 来源 |
|---|---:|---|
| 2D (128,128,8,8,8) | 22.21 | `full-ladder-rebench-v2-2026-09-26.md`，同一 session、同节点 gcp5-h100-0-15 |
| vectorized (128,128,8,8,8) | 32.71（RESULTS.md 写 32.73） | 同上 |
| 2D 自动调参 (128,128,16,16,8) | 33.78 | `RESULTS.md` |
| vectorized 自动调参 (128,128,16,8,8) | 33.49 | `RESULTS.md` |

- 默认参数下 +47% 是真的（同 session 同节点，但不是 ABAB 交替）。
- 线索：**调参后两者几乎一样**（33.78 vs 33.49）。说明 vectorized 的改动和"调参"能拿到的是差不多同一份收益（推测）。

### 6.1 SASS / ncu

还没做。

## 7. 问答记录

### 2026-10-03 第一轮

**Q：为什么要转置 A？是为了读的时候内存连续吗？**

A：对 ✅。2D 的 `As[128][8]` 里，一个 dotIdx 需要的 8 个 A 是竖着的一列，相邻相差 8 个 word，不连续，所以一个 dotIdx 内没法用向量读。转置成 `As[8][128]` 后，同一个 dotIdx 要的 8 个数在同一行里连续，可以 2 条 `LDS.128`。这是 Simon 转置的原因（他 A6000 的 SASS 里 2D 的 As 是 32 位 `LDS`）。

**Q：那不转置时，编译器怎么做到 `LDS.128` 的？**

A：换了个方向连续。dotIdx 循环（BK = 8）完全展开后，同一行 m 在 dotIdx 0..7 上的 `As[m][0..7]` 是连续 8 个 word。编译器一条 `LDS.128` 取 `As[m][0..3]`，供 dotIdx 0..3 用。条数和转置后一样（16 条/tile）。区别：2D 要**提前**把后面几个 dotIdx 的 A 读进寄存器放着；转置后每个 dotIdx 只读自己要的（推测：寄存器可能因此变少）。

**Q：`LDS.128` 和 global 的 float4 读是一回事吗？**

A：是同一个想法（一条指令 16B），不同内存空间（第 2.1 节）。global 那边 2D 没法自动合并，因为每个 thread 的 4 个 A 在不同行，不连续。

**Q：我们的实测真的变快了吗？提升从哪来？**

A：变快了：同 session 同节点 22.21 → 32.71T。来源还没验证，我的推测排序：

1. **寄存器 / occupancy**（最大嫌疑）：如果 vectorized 寄存器降到 ≤128 → 每 SM 2 个 block、16 个 warp，搬运延迟能被另一个 block 的计算盖住。理由：不用跨 dotIdx 提前读 A；float4 搬运少了地址计算。
2. 搬运阶段更短（LDG 8 → 2、边界判断变少），sync 之间的空等更短：有一点，但预计不大。
3. `STG.128` 写回：只发生一次，预计很小。

可证伪的预测：vectorized 的 SASS `-res-usage` 显示寄存器 ≤128（推测）。如果还是 160 左右，第 1 条就错了。

## 8. 我说错过 / 纠正过的地方

（暂无）

## 9. 仓库文档里的问题（未修改）

- `worklog.md` Step 6 "Why transpose matters here"：沿用"2D 的 As 读是 2-way 冲突"，并说 float4 之后这冲突变成 "30%+ of inner-loop latency"。我们 build 的 2D SASS 里 As 是 `LDS.128`（broadcast、无冲突，ncu 证实），这套解释不适用；"30%+" 没有测量出处。
- `worklog.md` Step 6 "Change 1" 表格："1 × 128B per scalar load" 和 "1 × 128B per scalar-equivalent" 两列写法含糊；同时把收益归因于 LSU 指令数，但 LDG 只占指令约 1.4%（第 5.1 节，推测）。
- `matmul_vectorized.cu` 第 1–7 行注释说转置是 "for coalesced SMEM loads"：smem 没有 coalescing 的说法（那是 global 的概念），应是"连续、可向量化 / 无 bank conflict"。
- `matmul_vectorized.cu` 第 63、83 行：`float4` 读要求地址 16B 对齐，N 不是 4 的倍数时会非对齐访问出错；边界分支只检查越界，不检查对齐。（N=4096 没问题。）

## 10. 要点 / 自测题

（待补）
