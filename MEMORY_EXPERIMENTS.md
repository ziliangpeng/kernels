# CUDA Memory Transfer Experiments

**Environment:** H100 80GB, PCIe Gen5 x16 (~64 GB/s theoretical), CUDA 12.4

---

## Experiment 1: Warmup Effects

**Question:** Does iteration count affect average transfer bandwidth?

**Setup:**
- Size: 100 MB
- Iterations: 1, 5, 10, 20
- Memory: Pageable (malloc) vs Pinned (cudaMallocHost)

### Results

| Iterations | Pageable H→D | Pageable D→H | Pinned H→D | Pinned D→H |
|------------|--------------|--------------|------------|------------|
| 1          | 9.37 GB/s    | 14.93 GB/s   | 25.60 GB/s | 26.56 GB/s |
| 5          | 10.29 GB/s   | 16.09 GB/s   | 25.70 GB/s | 26.58 GB/s |
| 10         | 11.32 GB/s   | 16.22 GB/s   | 25.71 GB/s | 26.59 GB/s |
| 20         | 11.52 GB/s   | 14.17 GB/s   | 25.72 GB/s | 26.53 GB/s |

**Improvement from 1st to 10th iteration:**
- Pageable H→D: +21%
- Pageable D→H: +9%
- Pinned H→D: +0.4% (negligible)
- Pinned D→H: +0.1% (negligible)

### Key Findings

1. **Pageable memory has warmup** - First transfer is 9-21% slower
   - Cause: TLB caching, staging buffer reuse, PCIe link state

2. **Pinned memory is consistent** - <1% variance across iterations
   - No warmup needed, direct DMA path

3. **Even warm, pageable is 2.2x slower** - Warmup helps but not enough
   - Warm pageable: ~11.5 GB/s
   - Pinned: ~26 GB/s

4. **Both miss PCIe peak** - Neither reaches 64 GB/s theoretical
   - Pageable: 14-18% efficiency
   - Pinned: 40% efficiency

### Takeaways

- **Benchmark with multiple iterations** - Single-shot underestimates pageable by 20%
- **Pinned still wins when warm** - 2-3x faster than warmed pageable
- **First transfer penalty is real** - Cold pageable is 2.7x slower than pinned
- **Use pinned for repeated transfers** - Warmup doesn't close the gap

---

*Last Updated: 2025-12-31*
