# WGMMA version guide (rung 9c) — every version is a first-class kernel

Each WGMMA version lives as its own registered kernel — same status as every
other rung in this repo:

    ./matmul_bench --method wgmma_v2   # (v2, v3, v4, v5, v5_1, v6, v7, v8)

Files: `../matmul_wgmma_v{2,3,4,5,5.1,6,7,8}.{h,cu}` (v1 was never verify-PASS and has no kernel file; see it in git history). The working-tree file
`../matmul_wgmma.cu` is the development head (currently = v7); version
snapshots in this folder are retired — the git history of each kernel file
carries the full blame trail. Narrative worklogs: `../wgmma-v{2,4,5,6,7,8}-*.md`.

## Per-version notes (what to look at when reading the code)

**v1** — read to see how many traps the first attempt hits: wrong SBO
(LeetCUDA swizzle constants pasted onto no-swizzle), wrong epilogue mapping,
inexpressible row-pitch-32B layout, `+r` vs `+f` constraint, `__half4`.
Nothing here survived into v2 except the host wrapper.

**v2** — the reference implementation. Everything in it is sweep-verified:
no-swizzle K-major interleave (core matrix = 8x8 halfs = 128B contiguous;
atom (mi,ki) at mi*128 + ki*1024), LBO=1024/SBO=128, m64n64 accumulator
mapping (4 regs per 8-col group), and `fence.proxy.async.shared::cta`
between SMEM writes and wgmma issue. The 243-combo brute-force sweep
(`../wgmma_desc_sweep.cu`) is what found the only-passing descriptor combo.

**v3** — v2 + double-buffering with `wgmma.wait_group 1` (previous group
done -> its buffer reusable while current runs). Perf null: the 64x64 tile
is bandwidth-bound, latency hiding has nothing to feed. Also the version
where an undefined `wgmma_wait<1>` compiled to NOTHING because the build
script printed BUILD OK past a failed nvcc — bench silently ran the v2
binary. Lesson: always `rm .build/*.o` + grep the PTX for the new
instruction before believing a result.

**v4** — scale the CTA, not the macro: 128x128 tile, 4 warpgroups in a 2x2
quadrant split, each running the SAME v2 macro on its quadrant descriptor.
AI doubles (32 -> 64 FLOP/B). Two layout bugs fixed en route: quadrant
footprint is 2048B not 1024 (2 K-atoms x LBO), and the synch experiment had
its own buf/buf^1 aliasing bug.

**v5** — replace LDG+STS staging with `cp.async` 16B copies. Perf ~flat vs
v4: with 2 stages the copy only overlaps the tail of the wgmma. The point
of v5 is the negative result: **asynchrony alone buys nothing; depth buys
it.**

**v5.1** — same kernel, STAGES=4 (32KB SMEM), prologue fills 3 stages.
`cp.async.wait_group N` counting lesson: groups newer than tile i at wait
time are only i+1, i+2 (i+3 issues after the wait) — depth is min(2,
tilesLeft), clamped at the tail. Counting by stage-array size over-permits
and reads unlanded tiles.

**v7** — the controlled experiment that DISPROVES the bandwidth theory:
128x256 CTA (AI 85, roofline ~290T) landed flat vs v6. Two staging-offset
bugs en route (lchunk residue, +1024 vs +4096) share one root cause: changed
staging granularity, reused stale offset constants. Conclusion for the
road: remaining gap is barriers + issue rate, so the next levers are TMA
bulk copies and warp specialization, not bigger tiles.

**v6** — instruction-count reduction: one m64n128k16 per K-step per
warpgroup (2 wgs x 64x128 strips). The B operand's SMEM layout must follow
the macro's traversal: m64n128 walks 16 n-atoms x SBO=128 = 2048B per
k-chunk, so B atoms are (r/8)*128 + chunk*2048 with descriptor LBO=2048
(A keeps the 64-row quadrant layout, LBO=1024). Rule: **operand layout is a
property of the macro shape, not of the tile.**

**v8** — warp specialization, first contact: producer warpgroup issues all
staging (cp.async), two consumer warpgroups run m64n256k16; mbarrier
full/free pipeline replaces every steady-state __syncthreads. Correct but
deliberately serialized (consumer wgmma_wait<0> before releasing the
stage) — 150.52T vs v7's 163.4T is partly node spread, partly that
serialization. The 4-round bug chain (per-cycle phase math, by-value
parity param, branch-scoped __syncthreads deadlock, half-staged A) is the
warp-spec first-lesson set; see `../wgmma-v8-2026-09-29.md`.

**v9** — the TMA payoff: producer issues 6 TMA bulk instructions per 64-k
stage (vs 768 per-thread cp.async); SWIZZLE_128B TMA boxes write exactly
the layout the wgmma 128B-swizzle descriptor expects (sweep-verified
SBO=1024, maxrel 0.0). 305.02T — +87% over v6 in one rung. Bug chain:
dynamic-SMEM 1024B alignment, TMA x-coordinate hardcoded 0 (same slice
accumulated N/64 times), explicit scale-d operand.

## Roofline ladder so far

(v3 and v5 have no separate worklog — their results and lessons are in the
v2/v5.1 worklogs and the per-version notes above.)

64x64/1wg 104.4 -> 128x128/4wg 135.8 -> +cp.async-depth 142.4 -> m64n128
164.8. HBM cap at this tile ~230T (AI=64); cuBLAS 728.7T. Next: 128x256
CTA (AI 85), TMA, warp specialization.
