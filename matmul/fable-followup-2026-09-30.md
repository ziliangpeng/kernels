# Fable 5 follow-up: post-experiment review (2026-09-30)

Prompt: ~/tmp/fable_followup_prompt.txt. Verbatim archive below.

## Executive summary
- UNIFIED THEORY: v9.7 is at the NO-MULTICAST L2 BANDWIDTH ROOFLINE.
  Arithmetic: 48KB/lap ÷ 4.19 MFLOP/lap = 11.7 B/FLOP; at 515T that is
  ≈6.0 TB/s of L2→SMEM read traffic. H100 sustainable L2 read ≈5.5-7.5 TB/s.
  This single model explains all five flat/negative results.
- Hypothesis (c) (multicast dead post-swizzle) WRONG: multicast saves L2
  READ bandwidth (L2 reads once, cluster fabric distributes), not DRAM.
  B = 32/48 KB per lap; 2-CTA multicast cuts cluster traffic 96→64KB/lap.
- v9.12 lost to implementation tax, not direction: (a) 4→3 stage regression
  (fix: epilogue reuses drained pipeline SMEM, keep 4 stages = 192KB);
  (b) leader-serialized TMA issue (fix: SPLIT-ISSUE — each CTA issues half
  of B with mask 0x3, symmetric accounting); (c) pairing along M was right.
- v9.10 N=3072 deadlock prime suspect: cross-WG phase-merge ABA on
  free_bar count=2. Sequential epilogue ⇒ wg0 always exits first ⇒ wg0's
  arrive for tile i+1 mainloop lands while wg1 still owes an arrive for
  tile i's tail — same barrier, same stage, ambiguous phase. 96-lap tiles
  re-align (producer-paced); 48-lap tiles don't. 4096 surviving is a race
  win, not correctness. Fix: per-wg free_bar (count=1 each) or
  consumers-only boundary named-barrier.
- 4096 ANOMALY: per-lap 1.5µs @4096 vs 1.07µs @8192 (-40%/lap) — real,
  unexplained, not wave-tail (worth an ncu diff... blocked here) .
- Fair ceiling comparison is CUTLASS forced to 128x256 cluster 1x1
  (Fable prior: 600-650T @8192), NOT cuBLAS (which uses persistent
  ping-pong + cluster = different design point).
- v9.13 = fixed multicast expected +40~80T. If flat, THEN the design point
  is done and the arc can close honorably.

## Full response (verbatim)
