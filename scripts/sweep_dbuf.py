#!/usr/bin/env python3
"""Sweep warptile_dbuf configs on the current GPU.

Reads the valid-config list from gen_dbuf_dispatch.py (same source of truth
as the dispatch table), runs the benchmark binary once per config, records
TFLOPS incrementally (one CSV row per config, flushed immediately — safe
against preemption), and skips configs already present in the output CSV
(resumable: re-running continues where it stopped).

Usage:
  python3 scripts/sweep_dbuf.py [--binary ./matmul_bench] [--n 4096]
       [--top 10] [--tag an-h100-cluster]
"""
import argparse
import csv
import re
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dbuf_dispatch


def run_config(binary, n, cfg):
    BM, BN, BK, WM, WN, WNITER, TM, TN, NT = cfg
    cmd = [binary, "--method", "warptile_dbuf", "-n", str(n),
           "--dbuf-BM", str(BM), "--dbuf-BN", str(BN), "--dbuf-BK", str(BK),
           "--dbuf-WM", str(WM), "--dbuf-WN", str(WN),
           "--dbuf-WNITER", str(WNITER), "--dbuf-TM", str(TM),
           "--dbuf-TN", str(TN), "--dbuf-NT", str(NT)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"Performance: ([\d.]+) GFLOPS", out.stdout)
    if not m:
        return None
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./matmul_bench")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tag", default="h100")
    ap.add_argument("--outdir", default="matmul")
    args = ap.parse_args()

    configs = gen_dbuf_dispatch.all_configs()
    out_csv = os.path.join(args.outdir, f"dbuf-sweep-{args.tag}.csv")

    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.reader(f):
                if row and row[0] != "BM":
                    done.add(tuple(row[:-1]))
    todo = [c for c in configs if tuple(map(str, c)) not in done]
    print(f"{len(configs)} valid configs, {len(done)} already done, "
          f"{len(todo)} to run; N={args.n}", flush=True)

    results = []
    for i, cfg in enumerate(todo):
        gf = run_config(args.binary, args.n, cfg)
        BM, BN, BK, WM, WN, WNITER, TM, TN, NT = cfg
        if gf is None:
            print(f"[{i+1}/{len(todo)}] {cfg} FAILED/unsupported", flush=True)
            continue
        results.append((BM, BN, BK, WM, WN, WNITER, TM, TN, NT, gf))
        # incremental flush: one append per config, preemption-safe
        new_file = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["BM", "BN", "BK", "WM", "WN", "WNITER", "TM", "TN",
                            "NT", "GFLOPS"])
            w.writerow([BM, BN, BK, WM, WN, WNITER, TM, TN, NT, gf])
        print(f"[{i+1}/{len(todo)}] BM={BM} BN={BN} BK={BK} WM={WM} WN={WN} "
              f"WNITER={WNITER} TM={TM} TN={TN} NT={NT}: {gf/1000:.2f} TFLOPS",
              flush=True)

    # summary from the full CSV
    all_rows = []
    with open(out_csv) as f:
        for row in csv.reader(f):
            if row and row[0] != "BM":
                all_rows.append((tuple(map(int, row[:-1])), float(row[-1])))
    all_rows.sort(key=lambda r: -r[1])
    print(f"\nresults -> {out_csv} ({len(all_rows)} rows)")
    print(f"\nTOP {args.top}:")
    for cfg, gf in all_rows[:args.top]:
        print(f"  {cfg}: {gf/1000:.2f} TFLOPS")


if __name__ == "__main__":
    main()
