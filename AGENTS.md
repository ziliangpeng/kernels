# AGENTS.md

## Core rule: do only as asked

This is a learning project. The user is learning CUDA kernel optimization.
Do NOT read files, run benchmarks, or start implementing anything without
explicit instruction. Ask before acting. One step at a time.

## Reply language

Cantonese colloquial. Technical terms in English are fine.

## Where reports go

Kernel-related reports (benchmark results, ncu/nsys analyses, SASS dumps,
investigations) go INSIDE this repo, not in `~/reports/`:

- Put them next to the kernels they are about, e.g. matmul reports in `matmul/`.
- Name them `<topic>-YYYY-MM-DD.md` (e.g. `matmul/naive-blk-ncu-2026-10-01.md`).
- Commit and push to main after writing or updating a report.
