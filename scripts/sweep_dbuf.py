#!/usr/bin/env python3
"""Sweep warptile_dbuf configs on the current GPU.

Reads the valid-config list from gen_dbuf_dispatch.py --list (same source of
truth as the dispatch table), runs the benchmark binary once per config,
records TFLOPS, and writes results to matmul/dbuf-sweep-results-<tag>.csv
(completed in the repo, per the "perf data lives in the repo" rule).

Usage:
  python3 scripts/sweep_dbuf.py [--binary ./matmul_bench] [--n 4096]
       [--iters 100] [--top 10] [--tag gcp5-h100]
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
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
    print(f"{len(configs)} valid configs; N={args.n}")

    results = []
    for i, cfg in enumerate(configs):
        gf = run_config(args.binary, args.n, cfg)
        BM, BN, BK, WM, WN, WNITER, TM, TN, NT = cfg
        if gf is None:
            print(f"[{i+1}/{len(configs)}] {cfg} FAILED/unsupported")
            continue
        results.append((BM, BN, BK, WM, WN, WNITER, TM, TN, NT, gf))
        print(f"[{i+1}/{len(configs)}] BM={BM} BN={BN} BK={BK} WM={WM} WN={WN} "
              f"WNITER={WNITER} TM={TM} TN={TN} NT={NT}: {gf/1000:.2f} TFLOPS")

    results.sort(key=lambda r: -r[-1])
    out_csv = os.path.join(args.outdir, f"dbuf-sweep-{args.tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["BM", "BN", "BK", "WM", "WN", "WNITER", "TM", "TN", "NT", "GFLOPS"])
        w.writerows(results)
    print(f"\nresults -> {out_csv}")
    print(f"\nTOP {args.top}:")
    for r in results[:args.top]:
        print(f"  {r[:-1]}: {r[-1]/1000:.2f} TFLOPS")


if __name__ == "__main__":
    main()
