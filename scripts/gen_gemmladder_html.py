#!/usr/bin/env python3
"""Generate GEMM ladder HTML report (index + per-kernel detail pages) in ~/code/kernels.

Output: docs/gemm-ladder/index.html + docs/gemm-ladder/kernels/<slug>.html
Each detail page: description & improvement → full kernel code → perf data → GitHub-style diff vs parent.
"""
import difflib
import html
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAT = REPO / "matmul"
OUT = REPO / "docs" / "gemm-ladder"
KO = OUT / "kernels"

# ---------------- perf data (RESULTS-POSTFIX-2026-10-02.md, job 221814) ----------------
P4096 = {"wmma": "28.1", "v2": "120.3", "v3": "115.4", "v4": "164.8", "v5": "157.2",
         "v5_1": "175.5", "v6": "208.3", "v7": "207.5", "v8": "187.0", "v8_1": "178.9",
         "v9": "505.7", "v9_1": "377.3", "v9_2": "522.4", "v9_3": "514.8", "v9_4": "526.6",
         "v9_5": "581.5", "v9_6": "597.2", "v9_7": "696.2", "v9_8": "704.5", "v9_9": "700.8",
         "v9_10": "543.1", "v9_11": "713.1", "v9_12": "548.7", "v9_13": "681.0",
         "v9_14": "558.7", "v9_15": "648.0", "cublas": "724.7"}
P8192 = {"wmma": "29.3", "v2": "122.3", "v3": "120.5", "v4": "173.4", "v5": "164.3",
         "v5_1": "187.0", "v6": "225.8", "v7": "221.1", "v8": "200.4", "v8_1": "192.0",
         "v9": "542.4", "v9_1": "388.6", "v9_2": "561.3", "v9_3": "590.1", "v9_4": "559.0",
         "v9_5": "604.1", "v9_6": "681.9", "v9_7": "713.9", "v9_8": "697.1", "v9_9": "742.2",
         "v9_10": "569.3", "v9_11": "711.9", "v9_12": "580.3", "v9_13": "688.4",
         "v9_14": "555.0", "v9_15": "660.4", "cublas": "739.7"}
KO8192 = {"v9_11": "746.2"}  # nsys kernel-only median (RESULTS-KERNELONLY-2026-10-01.md)
SIMT4096 = {"naive": "5.29", "coalesced": "5.73", "smem": "8.99", "1d": "19.36",
            "2d": "33.78", "vectorized": "36.47", "warptile": "37.71", "dbuf": "37.60"}
SIMTF16 = {"smem": "9.36", "vectorized": "36.47"}  # fp16 column where it differs notably
CUBLAS_FP32_4096 = "51.93"

# NCU @8192 (NCU-PROFILING-2026-10-02.md): tensor% / mem% / L2% / occupancy% / verdict
NCU = {
    "wmma":  ("4.4",  "98.9", "19.4", "74.4", "pure DRAM-bound"),
    "v2":    ("12.8", "96.5", "79.5", "49.4", "mem-bound"),
    "v3":    ("12.0", "91.7", "57.4", "49.5", "mem-bound"),
    "v4":    ("18.3", "84.4", "51.5", "49.3", "mem-bound"),
    "v5":    ("17.1", "82.9", "50.5", "48.8", "mem-bound"),
    "v5_1":  ("18.6", "90.6", "62.1", "49.7", "mem-bound"),
    "v6":    ("22.9", "91.5", "71.9", "24.7", "mem-bound"),
    "v7":    ("23.3", "85.9", "53.4", "12.5", "mem-bound"),
    "v8":    ("20.9", "82.4", "47.8", "18.5", "mem-bound"),
    "v8_1":  ("20.5", "80.6", "45.6", "18.5", "mem-bound"),
    "v9":    ("67.1", "60.5", "71.4", "13.8", "phase transition (TMA)"),
    "v9_1":  ("47.4", "60.1", "74.5", "13.9", "slower v9 (geometry)"),
    "v9_2":  ("67.3", "61.8", "71.7", "13.8", "= v9"),
    "v9_3":  ("69.8", "52.3", "54.1", "13.8", "band swizzle"),
    "v9_4":  ("68.1", "46.8", "63.1", "13.8", "negative variant"),
    "v9_5":  ("70.9", "51.7", "51.8", "18.3", "TMA epilogue"),
    "v9_6":  ("82.9", "64.6", "64.6", "13.7", "4-stage"),
    "v9_7":  ("88.5", "58.6", "58.6", "18.2", "tensor-pipe-bound"),
    "v9_8":  ("86.2", "56.6", "54.8", "18.8", "tensor-pipe-bound"),
    "v9_9":  ("88.6", "60.8", "60.2", "18.2", "tensor-pipe-bound"),
    "v9_10": ("64.7", "42.5", "37.2", "14.1", "256x256 fail (2-stage)"),
    "v9_11": ("89.1", "66.6", "67.8", "18.2", "tensor-pipe-bound (champ)"),
    "v9_12": ("67.9", "46.4", "46.6", "18.3", "multicast flat (impl tax)"),
    "v9_13": ("86.2", "59.1", "59.2", "18.2", "multicast fixed, still flat"),
    "v9_14": ("69.4", "71.3", "71.2", "9.4",  "issue-limited"),
    "v9_15": ("84.4", "70.7", "70.6", "—",    "warp-recovered, still -9%"),
}
STALLS = {  # WarpState deep-dive @8192 (per-issue-slot cycles)
    "v6":    "long_scoreboard 17.7 ≫ barrier 3.2 — 等 SMEM/L2 load,典型 memory-bound",
    "v9_3":  "barrier 5.4, long_scoreboard 4.2, lg_throttle 2.8;L1 hit 76%",
    "v9_6":  "barrier 6.4, long_scoreboard 1.1 — 4-stage 把 scoreboard stall 修掉了",
    "v9_7":  "barrier 10.1 主导 — mbarrier 节拍;long_scoreboard ~1",
    "v9_9":  "barrier 11.3 主导;long_scoreboard ~1",
    "v9_11": "barrier 9.3 主导;long_scoreboard ~1",
    "v9_10": "barrier 4.5 + wait 3.0 + long_scoreboard 2.8 分散 — 延迟没藏住",
    "v9_12": "barrier 9.6 + long_scoreboard 3.2 — split-issue 串行化可见",
    "v9_13": "barrier 9.8 + long_scoreboard 0.9 — 实现税已除,multicast 本身无增益",
    "v9_14": "barrier 3.87, wait 1.51, selected 1.00, long_scoreboard 0.95, gmma 0.93",
    "v9_15": "barrier 6.25, wait 2.33, long_scoreboard 1.42, gmma 1.02",
}

# (slug, rank, display name, code stem, diff parent stem or None)
KERNELS = [
    ("naive",      1,  "naive",         "matmul_naive",         None),
    ("coalesced",  2,  "coalesced",     "matmul_coalesced",     "matmul_naive"),
    ("smem",       3,  "smem_tiling",   "matmul_smem",          "matmul_coalesced"),
    ("1d",         4,  "1d_blocktile",  "matmul_1d_blocktile",  "matmul_smem"),
    ("2d",         5,  "2d_blocktile",  "matmul_2d_blocktile",  "matmul_1d_blocktile"),
    ("vectorized", 6,  "vectorized",    "matmul_vectorized",    "matmul_2d_blocktile"),
    ("warptile",   7,  "warptile",      "matmul_warptile",      "matmul_vectorized"),
    ("dbuf",       8,  "warptile_dbuf", "matmul_warptile_dbuf", "matmul_warptile"),
    ("wmma",       9,  "wmma",          "matmul_wmma",          None),
    ("v2",         10, "wgmma_v2",      "matmul_wgmma_v2",      None),
    ("v3",         11, "wgmma_v3",      "matmul_wgmma_v3",      "matmul_wgmma_v2"),
    ("v4",         12, "wgmma_v4",      "matmul_wgmma_v4",      "matmul_wgmma_v3"),
    ("v5",         13, "wgmma_v5",      "matmul_wgmma_v5",      "matmul_wgmma_v4"),
    ("v5_1",       14, "wgmma_v5_1",    "matmul_wgmma_v5.1",    "matmul_wgmma_v5"),
    ("v6",         15, "wgmma_v6",      "matmul_wgmma_v6",      "matmul_wgmma_v5.1"),
    ("v7",         16, "wgmma_v7",      "matmul_wgmma_v7",      "matmul_wgmma_v6"),
    ("v8",         17, "wgmma_v8",      "matmul_wgmma_v8",      "matmul_wgmma_v7"),
    ("v8_1",       18, "wgmma_v8_1",    "matmul_wgmma_v8_1",    "matmul_wgmma_v8"),
    ("v9",         19, "wgmma_v9",      "matmul_wgmma_v9",      "matmul_wgmma_v8"),
    ("v9_1",       20, "wgmma_v9_1",    "matmul_wgmma_v9_1",    "matmul_wgmma_v9"),
    ("v9_2",       21, "wgmma_v9_2",    "matmul_wgmma_v9_2",    "matmul_wgmma_v9"),
    ("v9_3",       22, "wgmma_v9_3",    "matmul_wgmma_v9_3",    "matmul_wgmma_v9_2"),
    ("v9_4",       23, "wgmma_v9_4",    "matmul_wgmma_v9_4",    "matmul_wgmma_v9_3"),
    ("v9_5",       24, "wgmma_v9_5",    "matmul_wgmma_v9_5",    "matmul_wgmma_v9_3"),
    ("v9_6",       25, "wgmma_v9_6",    "matmul_wgmma_v9_6",    "matmul_wgmma_v9_5"),
    ("v9_7",       26, "wgmma_v9_7",    "matmul_wgmma_v9_7",    "matmul_wgmma_v9_5"),
    ("v9_8",       27, "wgmma_v9_8",    "matmul_wgmma_v9_8",    "matmul_wgmma_v9_7"),
    ("v9_9",       28, "wgmma_v9_9",    "matmul_wgmma_v9_9",    "matmul_wgmma_v9_7"),
    ("v9_10",      29, "wgmma_v9_10",   "matmul_wgmma_v9_10",   "matmul_wgmma_v9_8"),
    ("v9_11",      30, "wgmma_v9_11",   "matmul_wgmma_v9_11",   "matmul_wgmma_v9_7"),
    ("v9_12",      31, "wgmma_v9_12",   "matmul_wgmma_v9_12",   "matmul_wgmma_v9_11"),
    ("v9_13",      32, "wgmma_v9_13",   "matmul_wgmma_v9_13",   "matmul_wgmma_v9_12"),
    ("v9_14",      33, "wgmma_v9_14",   "matmul_wgmma_v9_14",   "matmul_wgmma_v9_13"),
    ("v9_15",      34, "wgmma_v9_15",   "matmul_wgmma_v9_15",   "matmul_wgmma_v9_14"),
]

# per-kernel narrative (impl / delta / verdict), zh with English terms
N = {
"naive": ("一个线程算一个 C 元素:对 A 的一整行和 B 的一整列做 dot product,直接从 global memory 读。所有访问都是不 coalesce 的 4 字节 load,arithmetic intensity 约 0.25 FLOP/byte。",
 "基线 kernel,无上一级。",
 "峰值 989T 的 0.5% —— 构造上就是 memory-bound,作为正确性参照物存在。"),
"coalesced": ("还是一线程一元素,但 warp 里的线程沿 K 方向读连续地址,凑成 128B 的 coalesced 事务。",
 "只改内存访问布局:每 warp 的 <code>float</code> load 变连续;计算逻辑不变。",
 "+8% —— kernel 是 latency-bound(每线程串行走 K),事务效率提高但请求数一个没少。教训:coalescing ≠ 减少请求。"),
"smem": ("每个 block 算 C 的一个 tile,把 A/B tile 碎片 stage 进 shared memory,整个 block 复用同一份碎片;SMEM 加 padding 消 bank conflict。",
 "数据复用:global load 摊到整个 tile 上。AI 翻倍。",
 "+57% —— 复用是第一个真正的乘数。FP16 变体 9.36T,首次 FP16 &gt; FP32(带宽受限区)。"),
"1d": ("沿一个维度做 register 级复用:每线程累加一列多个 C 元素,部分和驻留寄存器。",
 "在 smem_tiling 上加 register tiling,每线程工作量 1 → 一条。",
 "+115% —— 寄存器复用是最便宜的乘数,不用加 SMEM 就翻 AI。"),
"2d": ("每线程拥有 C 的二维子 tile(4×4):A 碎片喂 4 列、B 碎片喂 4 行,两个轴都复用。",
 "第二维 register 复用,每线程输出从一条长成方阵。",
 "+74% —— AI 随两个复用轴乘积增长,每 FLOP load 再砍一半。"),
"vectorized": ("<code>float4</code>/<code>half2</code> 宽 load:16B 事务替代 4/2B。",
 "在 2d_blocktile 上只改事务宽度,每字节指令数降 4-8 倍。",
 "FP32 −1% 持平,FP16 +10% —— FP16 每元素字节减半,宽 load 恢复满事务宽度;FP16 成为此阶段冠军。"),
"warptile": ("Warp 级 tiling:warp 拥有 64×64 输出 tile,每 lane 拿 8×8 寄存器子 tile;autotuner 扫 (warp_m, warp_n, k)。",
 "复用从线程级升到 warp 级:每 FLOP 的 SMEM 带宽降约 8 倍。",
 "+12% (FP32) —— 逼近 SIMT roofline;NCU 显示下一瓶颈是 SMEM 带宽,标量代码治不了。硬编码 tuned 变体 39.07T。"),
"dbuf": ("warptile + <code>cp.async</code> double buffering:数学消费 stage 0 的同时 copy engine 填 stage 1。",
 "加异步 copy pipeline 做 latency hiding(Ampere 特性)。",
 "−0.3% 持平 —— 单变量负结果:AI 这么低时是 SMEM-bandwidth-bound,async 造不出带宽。教训:async 本身不买账,深度才买账。"),
"wmma": ("第一次用 tensor core:<code>wmma</code> API,16×16×16 fragment,<code>load_matrix_sync</code>/<code>mma_sync</code>/<code>store_matrix_sync</code>,FP16 输入 FP32 累加。",
 "计算从 FMA 换 tensor core;无 pipeline、无 swizzle。",
 "28.1T &lt; SIMT 39.1T —— 裸 tensor core 没管线打不过精调 SIMT。NCU:mem 98.9% / tensor 4.4%,fragment load 主导,MMA 延迟完全没藏。"),
"v2": ("Hopper WGMMA(<code>wgmma.mma_async</code> m64n64k16)用 64 位 SMEM descriptor 直读操作数;64×64 CTA 单 warpgroup;descriptor 常量(LBO/SBO/swizzle)由 243 组合暴力扫出。",
 "从 wmma 换代:warpgroup 级异步 MMA + SMEM 直读(不经寄存器 fragment),每指令 4× FLOP。",
 "+328% —— WGMMA 异步模型拆掉 fragment-load 墙。NCU:64×64 下仍 mem-bound(96.5%),tile 太小喂不饱数学单元。"),
"v3": ("v2 + double-buffer WGMMA,<code>wgmma.wait_group 1</code>:上一组还在跑就发下一组。",
 "指令级异步深度实验。",
 "−4% 持平 —— null result:64×64 bandwidth-bound,latency hiding 无米下锅。build 教训:<code>wgmma_wait&lt;1&gt;</code> 未定义时 nvcc 失败后脚本仍打 BUILD OK,bench 静默跑旧 binary。"),
"v4": ("放大 CTA 而非 macro:128×128 tile,4 个 warpgroup 做 2×2 象限切分,各自跑 v2 macro。",
 "tile 64×64 → 128×128,AI 32 → 64 FLOP/B。",
 "+43% —— 每 stage 字节复用更多,memory-bound 天花板随 tile 上移。途中修两个布局 bug(象限 footprint 2048B、synch 实验 buf 别名)。"),
"v5": ("LDG+STS 搬运换成 <code>cp.async</code> 16B copy。",
 "只换 load 机制;tile 与数学不变。",
 "−5% 持平 —— 对照实验证明 async 本身不买账:2-stage 下 copy 只 overlap 到 WGMMA 尾巴。与 v5_1 构成阶梯最干净的单变量对。"),
"v5_1": ("与 v5 完全相同,STAGES=4(32KB):prologue 填 3 stage,稳态 3 个在飞。",
 "仅 pipeline 深度 2 → 4。",
 "比 v5 +12% / v4 +18% —— 深度把 async copy 变成真 overlap。此处习得 <code>cp.async.wait_group N</code> 计数语义(尾部深度 = min(2, tilesLeft))。"),
"v6": ("砍指令数:每 warpgroup 每 K-step 一条 <code>m64n128k16</code>(2 WG × 64×128 条带);B 的 SMEM 布局跟随 macro 遍历(16 n-atom × SBO)。",
 "更大 WGMMA macro:每指令 2× FLOP,发射流量减半。",
 "+19% —— 每 FLOP 指令更少直接降发射压力。规则:操作数布局是 macro 形状的属性,不是 tile 的。NCU 仍 mem-bound 91.5%。"),
"v7": ("128×256 CTA(AI 85,roofline ≈290T),沿用 v6 macro 结构。",
 "tile 128×128 → 128×256,AI 64 → 85。",
 "−0.4% 持平 —— 证伪「更大 tile」:多出的复用花不出去,barrier + 发射率先绑定。结论:下一杠杆是 TMA 和 warp specialization,不是更大 tile。"),
"v8": ("Warp specialization 初体验:1 个 producer warpgroup 包办 staging(cp.async),2 个 consumer warpgroup 跑 <code>m64n256k16</code>;mbarrier full/free 环形 pipeline 取代稳态 <code>__syncthreads</code>。",
 "角色分工 + mbarrier 环,替代 block 级 barrier。",
 "−10% vs v6 —— 有意串行化(consumer 在释放 stage 前 <code>wgmma_wait&lt;0&gt;</code>)保协议显然正确;为 v9 的结构铺路。四轮 bug 链是 warp-spec 第一课。"),
"v8_1": ("v8 的协议修正版(mbarrier 到达计数与 parity 语义修干净)。",
 "协议修复,无新机制。",
 "−14% vs v6 —— 同样是铺路 rung,数值略低于 v8。"),
"v9": ("TMA bulk copy:producer 每个 64-K stage 只发 <b>6 条 TMA 指令</b>(对比 cp.async 的 768 条逐线程);SWIZZLE_128B TMA box 直接写出 WGMMA descriptor 期望的布局(SBO=1024 扫描验证);128×256 CTA,2-stage,warp-specialized。",
 "换 load 路径:逐线程 cp.async → 每 CTA 的 TMA bulk tensor copy + 硬件 swizzle;一条指令搬一整个 tile stage。",
 "<b>一级跳 +87%</b> —— 阶梯最大单级跳。TMA 卸载内存路径(发射槽、LSU),tensor pipe 首次被喂饱:NCU 从 mem-bound(80-99%)翻转为 67% tensor / 60% mem,相变点。bug 链:动态 SMEM 1024B 对齐、TMA x 坐标硬编码 0、显式 scale-d。"),
"v9_1": ("v9 + consumer overlap + 3-stage,但 CTA 换 128×128。",
 "几何探针:tile 128×256 → 128×128 + 加深 pipeline。",
 "−25% —— 负结果几何对照:AI 85→64 的损失超过 3-stage 深度收益。Fable 曾标记为最大未解释断层,现已归因。L2 74.5% 为全家族最重。"),
"v9_2": ("v9 形状 + 3-stage 144KB + consumer WGMMA overlap 恢复。",
 "深度 2 → 3;此处发现 H100 per-block opt-in SMEM = 227KB(非 100KB)。",
 "+3% —— 128×256 形状在 2-stage 已近饱和,深度不是此时的剩余瓶颈。"),
"v9_3": ("v9_2 + band CTA swizzle(G=8):按 8 行一组发射 CTA,让并发 CTA 在 L2 共享 B 列。",
 "只改光栅化顺序:同 kernel、同 pipeline、不同 CTA→tile 映射。",
 "@4096 持平,但 @8192 +10% —— N-sweep 揭示:工作集在 L2 内时 swizzle 无事可做;规模上去是真实收益。教训:光栅化顺序在大规模下有影响。"),
"v9_4": ("一维 band 换成 8×2 二维矩形 CTA swizzle。",
 "swizzle 形状:长条 → 窄两列矩形。",
 "@8192 −11% —— 窄矩形伤 B 复用(两列 band 在复用前被逐出)。小 N verify FAIL 是 ragged-N 双射 bug,趋势已负不修。"),
"v9_5": ("TMA epilogue:accumulator 先 stage 进 SMEM,再由单线程发 2 条 TMA store bulk 写回。",
 "C 写回路径:逐线程 8B 散射 store → SMEM staging + bulk tensor store。",
 "@4096 +22T / @8192 +36T —— C 写放大是真的:散射 store 在尾部偷发射槽和 LSU 带宽。"),
"v9_6": ("4-stage 192KB pipeline(保持 v9_5 形状):TMA 预取跑在 WGMMA 排空更前面。",
 "深度 3 → 4,SMEM 144KB → 192KB。",
 "@8192 +78T —— epilogue 不堵尾部后,深度终于把 TMA 延迟余量换成吞吐。NCU long_scoreboard 4.2 → 1.1。"),
"v9_7": ("<b>v9_5 + v9_6 合体</b>:TMA epilogue AND 4-stage 叠在 v9_3 band-swizzle 形状上。128×256,384 线程,2 consumer WG 跑 m64n256k16。",
 "两个独立增益复合:实测 +57.6T 超过加法预测 +49.3T。",
 "<b>@4096 达 cuBLAS 96%</b> —— 协同机制:epilogue 释放发射槽,深度让 TMA 填充更靠前,互拆对方天花板。首个 tensor pipe 饱和 kernel(88.5%)。"),
"v9_8": ("Persistent CTA:grid = 132 CTA(每 SM 一个)循环领 tile;外加不节流 WGMMA 发射纪律。",
 "调度模型:硬件 CTA 轮换 → 软件持久循环。",
 "持平 —— wave-quantization 假设证伪:消除 N=4096 的 3.88-wave 尾巴无收益(启动开销早被 64 个 K-lap 摊薄);大 N 每 tile 边界 pipeline 排空反而 −6T。"),
"v9_9": ("<code>setmaxnreg</code> 寄存器再分配:producer warpgroup 让出寄存器给 consumer(CUTLASS 惯例)。",
 "只改寄存器分配。",
 "持平 —— Fable 的寄存器饥饿假设证伪:ptxas 0 spill / 154 regs,consumer 从未饿过。@8192 742.2T = 同 run cuBLAS 的 100%(统计平手)。"),
"v9_10": ("256×256 CTA + persistent cooperative + async epilogue。",
 "tile 翻倍至 256×256(SMEM 只容 2-stage),保持 persistent。",
 "−22% —— occupancy 崩塌(1 CTA/SM × 2 stage 藏不住延迟)压过 macro 收益;tensor 仅 64.7%,stall 分散(barrier 4.5 + wait 3.0 + scoreboard 2.8)。插曲:「N=3072 死锁」实为 CPU verify 超时误诊,协议本来就是对的。"),
"v9_11": ("v9_7 家族 + 不节流 WGMMA 发射(更深 commit group)+ swizzled-accumulator epilogue。",
 "放开发射纪律(不再每条指令 wait)+ 清理 epilogue 写模式。",
 "<b>kernel-only 冠军:708.0T @4096 / 746.2T @8192 = cuBLAS 的 98.7% / 95.8%</b> —— tensor 89.1%,顶在 design point 硬件极限;剩余 ~11% 是 WGMMA 组间 mbarrier 节拍。"),
"v9_12": ("第一个 2-CTA cluster + TMA multicast on B:leader 替两个 CTA 发 B(mask 0x3),mbarrier 台账用 <code>mapa</code> 远程 arrive;退到 3-stage(leader 串行化代价)。",
 "cluster 启动 + multicast load 路径。",
 "−23% —— 实现税:leader 串行发 TMA + 深度损失超过 multicast 省的。sm_90a cluster 第一课集:per-destination barrier 记账、remote-arrive-only、tid0-in-producer-branch epilogue 死代码。"),
"v9_13": ("修好的 multicast:split-issue(每 CTA 发自己的 A + 一半 B,均 multicast)、对称 barrier 台账(两 rank 记账一致)、恢复 4-stage、epilogue 前 cluster-drain 保护。",
 "发射对称 + 深度恢复 —— 清零 v9_12 的实现税。",
 "比 v9_12 +60T,仍比 v9_11 −4% —— 税收回来了,multicast 本身买零:NCU 显示 L2 仅 46-59%,128×256 下不是 L2-bound,省 L2 读买不到时间。「multicast 治的病在这个几何下不存在」。"),
"v9_14": ("multicast 搬进 CUTLASS 赢家几何:128×128 CTA,cluster 沿 M(pair 共享 B tile),6-stage 192KB,2 consumer WG 跑 m64n128k16,split-issue。",
 "几何 128×256 → 128×128:AI 减半、L2 流量翻倍 —— multicast 理应兑现的地方。",
 "−23% vs v9_11 —— issue-limited:m64n128k16 每指令 FLOP 减半,发射路径先于任何 pipe 饱和(tensor 69.4 / mem 71.3 双不饱和)。L2 的病没轮到发作。途中修 k32 bug(consumer 每 k64 lap 只消费 k32)。"),
"v9_15": ("256×128 高 tile:640 线程(1 producer + 4 consumer WG,各 m64n128k16),4-stage,cluster 沿 M + split-issue B multicast(AI 104.6,比冠军少 20% L2 读)。",
 "warp 容量翻倍(4 consumer WG)+ 每 SM 工作密度恢复。",
 "比 v9_14 +19%,仍比 v9_11 −9% —— warp 容量诊断证实(tensor 69.4 → 84.4%),但 v9_11 更简单结构(每指令 2× FLOP、一半 barrier 握手)还是赢;mem 70%,L2 节省仍兑不成时间。multicast 三连击收官。"),
}

CSS = """
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; --yellow:#d29922; --cyan:#39c5cf; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,"SF Pro Text","PingFang SC","Segoe UI",Roboto,sans-serif; line-height:1.6; padding:28px 16px 80px; }
  .wrap { max-width:1060px; margin:0 auto; }
  header { text-align:center; margin-bottom:24px; }
  h1 { font-size:26px; letter-spacing:-0.4px; }
  .subtitle { color:var(--muted); margin-top:5px; font-size:14px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; margin-bottom:14px; overflow:hidden; }
  .card-head { display:flex; align-items:baseline; gap:12px; padding:13px 18px 9px; flex-wrap:wrap; }
  .rank { font-family:ui-monospace,Menlo,monospace; color:var(--muted); font-size:13px; min-width:34px; }
  .kname { font-family:ui-monospace,Menlo,monospace; font-size:16px; font-weight:700; color:var(--accent); text-decoration:none; }
  .kname:hover { text-decoration:underline; }
  .tflops { margin-left:auto; font-family:ui-monospace,Menlo,monospace; font-size:14px; }
  .tflops b { font-size:17px; color:var(--green); }
  .delta { font-size:12px; color:var(--muted); }
  .phase { margin:30px 0 12px; display:flex; align-items:center; gap:12px; }
  .phase h2 { font-size:18px; white-space:nowrap; }
  .phase .line { flex:1; height:1px; background:var(--border); }
  .phase .tag { font-size:12px; color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:4px; }
  th { text-align:left; color:var(--muted); font-weight:600; padding:6px 8px; border-bottom:1px solid var(--border); font-size:11.5px; }
  td { padding:5px 8px; border-bottom:1px solid #21262d; font-family:ui-monospace,Menlo,monospace; }
  tr:hover td { background:#1c2129; }
  .pos { color:var(--green); } .neg { color:var(--red); } .flatc { color:var(--yellow); }
  .klink { color:var(--accent); text-decoration:none; } .klink:hover { text-decoration:underline; }
  .note { font-size:12px; color:var(--muted); margin-top:8px; }
  .pagehead { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:6px; }
  .pagehead h1 { font-size:22px; font-family:ui-monospace,Menlo,monospace; color:var(--accent); }
  .back { font-size:13px; color:var(--muted); text-decoration:none; } .back:hover { color:var(--accent); }
  .sec { margin-top:22px; }
  .sec h2 { font-size:16px; color:var(--cyan); margin-bottom:8px; border-bottom:1px solid var(--border); padding-bottom:5px; }
  .sec p { font-size:14px; margin-bottom:8px; }
  .twocol { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:760px) { .twocol { grid-template-columns:1fr; } }
  pre.code { background:#0a0d12; border:1px solid var(--border); border-radius:8px; padding:14px; overflow-x:auto; font-family:ui-monospace,Menlo,monospace; font-size:12px; line-height:1.45; max-height:560px; overflow-y:auto; }
  pre.diff { background:#0a0d12; border:1px solid var(--border); border-radius:8px; padding:14px; overflow-x:auto; font-family:ui-monospace,Menlo,monospace; font-size:12px; line-height:1.45; max-height:640px; overflow-y:auto; }
  .dl-add { color:#3fb950; background:rgba(63,185,80,0.12); display:block; }
  .dl-del { color:#f85149; background:rgba(248,81,73,0.12); display:block; }
  .dl-ctx { color:#8b949e; display:block; }
  .dl-hdr { color:#39c5cf; display:block; }
  .verdict { padding:10px 13px; border-radius:7px; font-size:13.5px; border:1px solid; }
  .v-win { background:rgba(63,185,80,.10); border-color:rgba(63,185,80,.35); } .v-flat { background:rgba(210,153,34,.10); border-color:rgba(210,153,34,.35); } .v-neg { background:rgba(248,81,73,.10); border-color:rgba(248,81,73,.35); }
  .chip { background:var(--panel); border:1px solid var(--border); border-radius:999px; padding:3px 11px; font-size:12px; color:var(--muted); }
  code { font-family:ui-monospace,Menlo,monospace; background:#21262d; padding:1px 5px; border-radius:4px; font-size:12.5px; color:#79c0ff; }
"""

def esc(s):
    return html.escape(s, quote=False)

def verdict_class(slug):
    if slug in ("naive",): return "v-flat"
    if slug in ("coalesced","smem","1d","2d","warptile","v2","v4","v5_1","v6","v9","v9_3","v9_5","v9_6","v9_7","v9_11","v9_15"): 
        return "v-win" if slug != "v9_15" else "v-flat"
    if slug in ("vectorized","dbuf","v3","v5","v7","v9_2","v9_8","v9_9","v9_13"):
        return "v-flat"
    return "v-neg"

def perf_rows(slug):
    rows = []
    if slug in P4096:
        rows.append(("bench 端到端 @4096(修税后)", P4096[slug] + " TFLOPS"))
        rows.append(("bench 端到端 @8192", P8192[slug] + " TFLOPS"))
        if slug in KO8192:
            rows.append(("nsys kernel-only @8192(中位)", KO8192[slug] + " TFLOPS"))
        rows.append(("占同 run cuBLAS @8192", f"{float(P8192[slug])/float(P8192['cublas'])*100:.0f}%"))
        rows.append(("占峰值 989T @8192", f"{float(P8192[slug])/989.4*100:.1f}%"))
    elif slug in SIMT4096:
        rows.append(("bench @4096(autotuned FP32)", SIMT4096[slug] + " TFLOPS"))
        if slug in SIMTF16 and slug != "vectorized":
            rows.append(("FP16 变体 @4096", SIMTF16[slug] + " TFLOPS"))
        rows.append(("占 cuBLAS FP32(51.93T)", f"{float(SIMT4096[slug])/float(CUBLAS_FP32_4096)*100:.1f}%"))
    return rows

def ncu_rows(slug):
    if slug not in NCU:
        return None
    t, m, l2, occ, verdict = NCU[slug]
    return [("tensor pipe SoL", t + "%"), ("memory(DRAM)SoL", m + "%"),
            ("L2 throughput", l2 + "%"), ("occupancy", occ + "%"), ("判定", verdict)]

def github_diff(a_lines, b_lines, a_name, b_name, ctx=2):
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    out = [f'<span class="dl-hdr">diff --git a/{esc(a_name)} b/{esc(b_name)}</span>']
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for ln in a_lines[i1:i2]:
                out.append(f'<span class="dl-ctx"> {esc(ln)}</span>')
        else:
            if tag in ("replace", "delete"):
                for ln in a_lines[i1:i2]:
                    out.append(f'<span class="dl-del">-{esc(ln)}</span>')
            if tag in ("replace", "insert"):
                for ln in b_lines[j1:j2]:
                    out.append(f'<span class="dl-add">+{esc(ln)}</span>')
    # collapse long runs of context
    collapsed, run = [], 0
    for ln in out:
        if "dl-ctx" in ln:
            run += 1
            if run <= ctx or run % 400 == 0:
                collapsed.append(ln)
            elif run == ctx + 1:
                collapsed.append('<span class="dl-ctx">   …(上下文省略)…</span>')
        else:
            run = 0
            collapsed.append(ln)
    return "\n".join(collapsed)

def read_code(stem):
    cu = MAT / f"{stem}.cu"
    h = MAT / f"{stem}.h"
    txt = f"// ===== {stem}.h =====\n" + h.read_text() + f"\n// ===== {stem}.cu =====\n" + cu.read_text()
    return txt

def phase_of(slug):
    if slug in ("naive","coalesced","smem","1d","2d","vectorized","warptile","dbuf"):
        return 1
    if slug in ("wmma","v2","v3","v4","v5","v5_1","v6","v7","v8","v8_1"):
        return 2
    if slug in ("v9","v9_1","v9_2","v9_3","v9_4","v9_5","v9_6","v9_7","v9_8","v9_9","v9_10","v9_11"):
        return 3
    return 4

PHASES = {
    1: ("第一阶段 · SIMT 标量 kernel", "FP32,每级一个优化"),
    2: ("第二阶段 · Tensor core —— TMA 之前", "WMMA → WGMMA,NCU 全程 mem-bound 80-99%"),
    3: ("第三阶段 · TMA 时代", "相变点:mem-bound → tensor-bound"),
    4: ("第四阶段 · Cluster multicast 弧线", "Fable +155T 假设的三连击验证"),
}

def gen():
    KO.mkdir(parents=True, exist_ok=True)
    for slug, rank, name, stem, parent in KERNELS:
        gen_kernel_page(slug, rank, name, stem, parent)
    gen_index()
    print(f"OK: {len(KERNELS)} kernel pages + index -> {OUT}")

def perf4096(slug):
    if slug in P4096: return P4096[slug]
    return SIMT4096.get(slug, "—")

def gen_kernel_page(slug, rank, name, stem, parent):
    impl, delta, verdict = N[slug]
    rows = perf_rows(slug)
    ncu = ncu_rows(slug)
    stall = STALLS.get(slug)
    code = read_code(stem)
    diff_html = ""
    if parent:
        a = read_code(parent).splitlines()
        b = read_code(stem).splitlines()
        diff_html = github_diff(a, b, parent + ".{h,cu}", stem + ".{h,cu}")
    ncu_html = ""
    if ncu:
        ncu_html = '<div class="sec"><h2>NCU profile @8192(锁频 1.43GHz)</h2><table>'
        for k, v in ncu:
            ncu_html += f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        ncu_html += "</table>"
        if stall:
            ncu_html += f'<p class="note" style="margin-top:8px">WarpState stalls / issue slot:{stall}</p>'
        ncu_html += "</div>"
    perf_html = '<div class="sec"><h2>性能数据</h2><table>'
    for k, v in rows:
        perf_html += f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
    perf_html += "</table>"
    if slug not in P4096 and slug not in SIMT4096:
        perf_html = ""
    diff_sec = ""
    if diff_html:
        diff_sec = f'''<div class="sec"><h2>Diff vs 上一代({esc(parent)})—— GitHub 风格</h2>
<pre class="diff">{diff_html}</pre>
<p class="note">绿 = 新增行,红 = 删除行,灰 = 未变上下文(长段省略)。diff 基准选择代码血统最近的父代。</p></div>'''
    page = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(name)} — GEMM 阶梯</title><style>{CSS}</style></head>
<body><div class="wrap">
<div class="pagehead"><a class="back" href="../index.html">← 返回阶梯</a></div>
<div class="pagehead"><h1>#{rank} {esc(name)}</h1><span class="delta">{esc(stem)}.{esc("h/cu")}</span></div>
<div class="verdict {verdict_class(slug)}" style="margin-top:10px">{verdict}</div>
<div class="sec"><h2>实现与改进</h2>
<div class="twocol">
<div><p><b>实现:</b>{impl}</p></div>
<div><p><b>与上一代的差异:</b>{delta}</p></div>
</div></div>
<div class="sec"><h2>Kernel 代码(完整)</h2>
<pre class="code">{esc(code)}</pre></div>
{perf_html}
{ncu_html}
{diff_sec}
<p class="note" style="margin-top:18px">数据源:matmul/RESULTS-POSTFIX-2026-10-02.md · matmul/NCU-PROFILING-2026-10-02.md · 代码:matmul/{esc(stem)}.cu</p>
</div></body></html>'''
    (KO / f"{slug}.html").write_text(page)

def gen_index():
    cards = []
    by_phase = {}
    for slug, rank, name, stem, parent in KERNELS:
        by_phase.setdefault(phase_of(slug), []).append((slug, rank, name))
    for ph in sorted(by_phase):
        title, tag = PHASES[ph]
        cards.append(f'<div class="phase"><h2>{title}</h2><div class="line"></div><div class="tag">{tag}</div></div>')
        for slug, rank, name in by_phase[ph]:
            p = perf4096(slug)
            v = N[slug][2]
            # short verdict: strip tags
            import re as _re
            v_short = _re.sub(r"<[^>]+>", "", v)[:110]
            cards.append(f'''<div class="card"><div class="card-head">
<span class="rank">#{rank}</span><a class="kname" href="kernels/{slug}.html">{esc(name)}</a>
<span class="tflops"><b>{p}</b> TFLOPS <span class="delta">@4096 → 详情</span></span></div></div>''')
    total = len(KERNELS)
    page = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>H100 GEMM 优化阶梯 — kernel 目录</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><h1>H100 GEMM 优化阶梯 — kernel 目录</h1>
<div class="subtitle">{total} 个 kernel,每级一个机制 —— 点名字进详情页(实现 → 代码 → 性能 → diff)</div></header>
{"".join(cards)}
<p class="note" style="text-align:center;margin-top:24px">冠军 wgmma_v9_11:708.0T @4096 / 746.2T @8192 kernel-only = cuBLAS 的 98.7% / 95.8% · 生成器 scripts/gen_gemmladder_html.py</p>
</div></body></html>'''
    (OUT / "index.html").write_text(page)

if __name__ == "__main__":
    gen()
