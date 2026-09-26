#!/usr/bin/env python3
"""Analyze a dbuf_autotune CSV: top configs, best-vs-default delta, dimension
trends (what the winner does differently)."""
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
DEFAULT = (128, 128, 16, 64, 64, 2, 8, 4, 128)

rows = []
with open(path) as f:
    for r in csv.reader(f):
        if not r or r[0] in ("BM", "") or r[0].startswith("#"):
            continue
        try:
            rows.append((tuple(map(int, r[:9])), float(r[9])))
        except (ValueError, IndexError):
            continue

if not rows:
    sys.exit("no rows")

rows.sort(key=lambda x: -x[1])
print(f"{len(rows)} configs measured\n")
print("TOP 10:")
for cfg, gf in rows[:10]:
    print(f"  {cfg}: {gf/1000:.2f} TFLOPS")

dflt = [gf for cfg, gf in rows if cfg == DEFAULT]
if dflt:
    best_cfg, best_gf = rows[0]
    print(f"\nDEFAULT {DEFAULT}: {dflt[0]/1000:.2f} TFLOPS")
    print(f"BEST    {best_cfg}: {best_gf/1000:.2f} TFLOPS  "
          f"(+{(best_gf/dflt[0]-1)*100:.1f}% vs default)")

# dimension trends among top 5%
n = max(5, len(rows) // 20)
top = [cfg for cfg, _ in rows[:n]]
names = ["BM", "BN", "BK", "WM", "WN", "WNITER", "TM", "TN", "NT"]
print(f"\nDIMENSION TRENDS among top {n} configs (min/median/max):")
for i, nm in enumerate(names):
    vals = sorted(c[i] for c in top)
    med = vals[len(vals)//2]
    print(f"  {nm:6s} {vals[0]} / {med} / {vals[-1]}")
