# WGMMA version index (rung 9c) — tutorial walk-through guide

One `.cu` snapshot per version, frozen at the commit where that version's
final fix landed. `git log --oneline -- matmul/matmul_wgmma.cu` has the full
blame trail; worklogs `../wgmma-v{2,4,5,6}-*.md` carry the measurement
narrative. Read bottom-up for the teaching arc.

| ver | file | first..last commit | shape | TFLOPS | what changed vs previous |
|---|---|---|---|---:|---|
| v1 | matmul_wgmma_v1.cu | 78a7b23..5bcab00 | m64n128k16, 1 wg, single-buffer | (never passed verify) | initial attempt: SS descriptors, device-transposed B |
| v2 | matmul_wgmma_v2.cu | bbf2d68..1c6ee60 | m64n64k16, 1 wg, single-buffer | 104.36 | sweep-verified shape; **fence.proxy.async** added (THE correctness fix) |
| v3 | matmul_wgmma_v3.cu | 8e55080..c6e5c5a | v2 + double-buffer + wait_group 1 | 100.64 | latency pipeline; null perf result (64-tile is HBM-bound) — plus the fake-BUILD-OK lesson |
| v4 | matmul_wgmma_v4.cu | e20c25a..26f8039 | 128x128 CTA, 4 wgs 2x2 quadrants, dbuf | 135.82 | reuse one m64n64 macro per quadrant; quadrant footprint = 2048B |
| v5 | matmul_wgmma_v5.cu | 16f4769..16f4769 | v4 + cp.async 16B, 2-stage | 130.04 | staging goes async; 2 stages too shallow to help |
| v5.1 | matmul_wgmma_v5.1.cu | 9373920..ec4644f | 4-stage pipeline, 3 copies in flight | 142.42 | depth (not async-ness) hides latency; wait-depth counting bug |
| v6 | matmul_wgmma_v6.cu | f1feb44..dbc4ed1 | m64n128k16, 2 wgs, 4-stage | **164.83** | wgmma instruction count halved; **B layout follows macro shape** (full-128-row atoms, LBO=2048) |
| v7 | matmul_wgmma_v7.cu | e0c18af..d1c95f4 | m64n256k16, 128x256 CTA, 3-stage | 162.73 | bigger tile bought NOTHING (AI 85 vs 64) -> v6's gap is latency/sync, not bandwidth; staging-granularity change without re-derived offsets = 2 bugs |

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

## Roofline ladder so far

64x64/1wg 104.4 -> 128x128/4wg 135.8 -> +cp.async-depth 142.4 -> m64n128
164.8. HBM cap at this tile ~230T (AI=64); cuBLAS 728.7T. Next: 128x256
CTA (AI 85), TMA, warp specialization.
