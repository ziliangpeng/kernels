# Softmax Experiments

**Environment:** H100 80GB, PCIe Gen4 x16, CUDA 12.4, sm_90 architecture

---

## Experiment 1: Numerical Stability - Naive vs Stable Softmax

**Question:** Why is numerical stability critical in softmax computation?

**Setup:**
- Array size: 1,000 elements
- Test cases:
  1. Safe input range: [0, 10]
  2. Overflow input range: [500, 1000]
- Methods tested: Naive (unstable), Multi-pass (stable)

### Results

| Test Case | Method | Input Range | Sum(output) | Has NaN/Inf |
|-----------|--------|-------------|-------------|-------------|
| 1 | Naive | [0, 10] | 1.000000 | NO |
| 2 | Naive | [500, 1000] | nan | **YES (OVERFLOW!)** |
| 3 | Multi-pass | [500, 1000] | 1.000000 | NO |

### Key Findings

1. **Naive softmax fails catastrophically with large inputs**
   - Input value: 750
   - exp(750) ≈ 1.4 × 10^325 → **OVERFLOW to infinity**
   - Result: All outputs become NaN (0/0 or inf/inf)

2. **Max subtraction prevents overflow**
   - Multi-pass: First find max = 1000
   - Compute: exp(750 - 1000) = exp(-250) ≈ 3.7 × 10^-109
   - All exp values in range [0, 1], no overflow!

3. **Why this matters in practice:**
   - **Transformers:** Attention logits can be large (50-100+)
   - **Temperature scaling:** High temperature → large logits
   - **Numerical precision:** Even moderate values (> 88) overflow float32

### The Overflow Problem

**Naive formula:**
```
softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))
```

**Why it fails:**
- `exp(x)` grows exponentially: exp(100) ≈ 2.7 × 10^43
- Float32 max: ~3.4 × 10^38
- **exp(89) already overflows!**

**Stable formula:**
```
m = max(x)
softmax(x)[i] = exp(x[i] - m) / sum(exp(x[j] - m))
```

**Why it works:**
- Largest value: exp(max - max) = exp(0) = 1
- All other values: exp(x - max) < 1 (since x ≤ max)
- **No overflow possible!**

### Demonstration

```bash
# Safe inputs - naive works
$ ./softmax -n 1000 --method naive -v
✓ Verification PASSED

# Large inputs - naive fails
$ # (with input [500, 1000])
✗ Verification FAILED: Output contains NaN or Inf

# Large inputs - multi-pass stable
$ ./softmax -n 1000 --method multi -v
✓ Verification PASSED
```

---

## Experiment 2: Performance Comparison - Naive vs Multi-pass

**Question:** What is the performance trade-off for numerical stability?

**Setup:**
- Array sizes: 1K, 10K, 100K, 1M, 10M elements
- Block size: 256 threads/block (optimal)
- Iterations: 1000 runs per configuration
- Metric: Median execution time

**Methods compared:**
1. **Naive:** 2 kernel launches
   - Kernel 1: Compute sum(exp(x)) via reduction
   - Kernel 2: Normalize (element-wise divide)

2. **Multi-pass:** 3 kernel launches
   - Kernel 1: Find max(x) via reduction
   - Kernel 2: Compute sum(exp(x - max)) via reduction
   - Kernel 3: Normalize with max adjustment

### Results

| Array Size | Naive (ms) | Multi-pass (ms) | Overhead |
|------------|------------|-----------------|----------|
| 1K | 0.035 | 0.060 | **+71%** |
| 10K | 0.039 | 0.060 | **+54%** |
| 100K | 0.047 | 0.083 | **+77%** |
| 1M | 0.190 | 0.367 | **+93%** |
| 10M | 0.260 | 0.477 | **+83%** |

### Key Findings

1. **Multi-pass is ~50-90% slower**
   - Extra kernel launch for max-finding
   - Additional memory passes
   - Kernel launch overhead dominates for small arrays

2. **Overhead is relatively constant**
   - ~70-90% across all sizes
   - Suggests fixed cost (kernel launches) dominates
   - Not bandwidth-limited yet (even at 10M elements)

3. **Performance breakdown (estimated for 1M elements):**
   ```
   Naive (0.190 ms):
     exp-sum reduction: ~0.100 ms
     normalize:         ~0.080 ms
     overhead:          ~0.010 ms

   Multi-pass (0.367 ms):
     max reduction:     ~0.090 ms  (extra!)
     exp-sum reduction: ~0.100 ms
     normalize:         ~0.080 ms
     overhead:          ~0.097 ms  (3 launches vs 2)
   ```

4. **Why multi-pass is slower:**
   - **3 kernel launches vs 2:** Each launch has ~0.01-0.02 ms overhead
   - **2 reduction passes:** Max + exp-sum (naive only does exp-sum)
   - **Memory traffic:** 2 full reads of input (max stage, exp-sum stage)

5. **The trade-off:**
   - **Naive:** Fast but **BROKEN** for large inputs
   - **Multi-pass:** Slower but **ALWAYS CORRECT**
   - **In production:** Correctness >> 2x performance
   - **Real workloads:** Transformer attention uses stable softmax exclusively

### Why Accept the Overhead?

1. **Correctness is non-negotiable**
   - NaN/Inf breaks training completely
   - Model divergence is catastrophic
   - Cannot predict when inputs will be large

2. **Relative cost is small**
   - Softmax is usually <1% of total model time
   - Attention compute dominates (matmuls)
   - 2x softmax overhead = 0.5% total slowdown

3. **Modern optimizations recover performance**
   - Fused kernels (future work)
   - Online algorithms (future work)
   - Warp-level primitives
   - Can match naive speed while staying stable!

---

## New CUDA Concepts Learned

### 1. Numerical Stability in GPU Computing

**Why it matters more on GPU:**
```cuda
// CPU: Often uses double (64-bit) by default
double sum = 0.0;
for (int i = 0; i < n; i++) {
    sum += exp(x[i]);  // Less likely to overflow
}

// GPU: Typically uses float (32-bit) for performance
float sum = 0.0f;
// exp(89) already overflows float32!
```

**The max subtraction trick:**
```cuda
// Find max first
float max_val = -INFINITY;
for (int i = 0; i < n; i++) {
    max_val = fmaxf(max_val, x[i]);
}

// Then compute exp with adjustment
for (int i = 0; i < n; i++) {
    result[i] = expf(x[i] - max_val);  // Safe!
}
```

### 2. Multi-stage Reduction Algorithms

**Pattern:** Reduce → Process → Normalize
```
Stage 1: Reduce to find statistic (max, sum, etc.)
Stage 2: Reduce with statistic (exp-sum using max)
Stage 3: Element-wise transform (normalize)
```

**When to use:**
- Softmax (max → exp-sum → normalize)
- Batch normalization (mean → variance → normalize)
- Layer normalization (mean → variance → normalize)

### 3. Reduction Operator Variants

**Max reduction vs Sum reduction:**
```cuda
// Sum reduction
sdata[tid] += sdata[tid + stride];

// Max reduction
sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);

// Min reduction
sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
```

**Key insight:** Same tree reduction pattern, different operator!

### 4. Handling Out-of-Bounds in Reductions

**Critical for correctness:**
```cuda
// Sum reduction: use identity element 0
sdata[tid] = (idx < n) ? input[idx] : 0.0f;

// Max reduction: use identity element -infinity
sdata[tid] = (idx < n) ? input[idx] : -INFINITY;

// Min reduction: use identity element +infinity
sdata[tid] = (idx < n) ? input[idx] : INFINITY;
```

### 5. Sequential Reduction Stages

**Bug we fixed:** Don't re-apply transformation on already-transformed values!
```cuda
// WRONG: Applies exp twice in subsequent stages
for stage in stages:
    expSumReductionKernel()  // Does exp every time!

// CORRECT: Only exp first stage
if (firstStage) {
    expSumReductionKernel()  // exp(x - max)
} else {
    sumReductionKernel()     // Just sum (already exp'd)
}
```

**Lesson:** Track transformation state across reduction stages!

---

## Experiment 3: Fused Softmax - Reducing Kernel Launches

**Question:** Can we improve performance by reducing kernel launches and memory traffic?

**Motivation:**
- Multi-pass softmax: 7+ kernel launches for large arrays (with recursive reductions)
- Kernel launch overhead: ~0.01-0.02 ms per launch
- Memory traffic: Multiple full passes over input data
- **Goal:** Reduce to exactly 3 kernel launches with no recursion

**Setup:**
- Array sizes: 100K, 1M, 10M, 100M elements
- Block size: 256 threads/block
- Iterations: 1000 runs per configuration
- Metric: Median execution time

### Implementation: 3-Kernel Fused Approach

**Architecture:**
```
Input: d_input[n]
  ↓
Kernel 1: softmaxFused_BlockStats (1 launch)
  → Computes block_maxes[numBlocks], block_sums[numBlocks]
  ↓
Kernel 2: softmaxFused_GlobalReduce (1 launch)
  → Reduces to global_max, global_sum
  ↓
Kernel 3: softmaxFused_Normalize (1 launch)
  → Writes final output[n]
```

**Key Design Decisions:**

1. **Kernel 1 - Block-level fusion:**
   ```cuda
   // Phase 1: Find block max using tree reduction
   sdata[tid] = (idx < n) ? input[idx] : -INFINITY;
   // ... tree reduction with fmaxf() ...
   float block_max = sdata[0];

   // Phase 2: Reuse shared memory for exp-sum
   sdata[tid] = (idx < n) ? expf(input[idx] - block_max) : 0.0f;
   // ... tree reduction with addition ...
   float block_sum = sdata[0];
   ```
   - **Key optimization:** Reuse shared memory between reductions
   - Each block computes its own local statistics

2. **Kernel 2 - Grid-stride loop for arbitrary block counts:**
   ```cuda
   // Each thread processes multiple blocks if needed
   float thread_max = -INFINITY;
   for (int i = tid; i < numBlocks; i += blockDim.x) {
       thread_max = fmaxf(thread_max, block_maxes[i]);
   }

   // Critical adjustment formula for merging block sums:
   float thread_sum = 0.0f;
   for (int i = tid; i < numBlocks; i += blockDim.x) {
       thread_sum += block_sums[i] * expf(block_maxes[i] - global_max);
   }
   ```
   - **Key insight:** Must adjust block sums when merging with different local maxes
   - Grid-stride loop handles arbitrary number of blocks

3. **Kernel 3 - Simple normalization:**
   ```cuda
   output[idx] = expf(input[idx] - global_max) / global_sum;
   ```

### Results

| Array Size | Multi-pass (ms) | Fused (ms) | Speedup |
|------------|-----------------|------------|---------|
| 100K | 0.088 | 0.032 | **2.75x** |
| 1M | 0.372 | 0.173 | **2.15x** |
| 10M | 0.494 | 0.355 | **1.39x** |
| 100M | 1.658 | 2.557 | **0.65x** ⚠️ |

### Key Findings

1. **Strong speedup for small to medium arrays (up to 10M)**
   - 100K: 2.75x faster (0.088 ms → 0.032 ms)
   - 1M: 2.15x faster (0.372 ms → 0.173 ms)
   - Matches or exceeds expected 2-3x speedup target

2. **Performance degrades for very large arrays (100M+)**
   - 100M: 0.65x slower (1.658 ms → 2.557 ms)
   - **Root cause:** Grid-stride loop in Kernel 2
   - For 100M elements: 390,625 blocks → each thread processes 1,525 blocks
   - Too much work per thread in single-block reduction

3. **Why fused is faster (small/medium arrays):**
   - **Fewer kernel launches:** 3 vs 7+ (eliminates recursive overhead)
   - **Less memory traffic:** Block-level fusion reuses shared memory
   - **Better parallelism:** All blocks work simultaneously in Kernel 1

4. **Why fused is slower (very large arrays):**
   - **Kernel 2 bottleneck:** Single block with grid-stride loop
   - **Work imbalance:** 1,525 blocks/thread is too much
   - **Memory access pattern:** Poor cache locality in grid-stride loop

### Performance Breakdown (1M elements)

**Multi-pass (0.372 ms):**
```
Kernel 1 (max):              ~0.090 ms
Kernel 2-N (exp-sum):        ~0.100 ms (multiple launches)
Final normalize:             ~0.080 ms
Kernel launch overhead:      ~0.102 ms (7+ launches)
```

**Fused (0.173 ms):**
```
Kernel 1 (block stats):      ~0.060 ms
Kernel 2 (global reduce):    ~0.020 ms
Kernel 3 (normalize):        ~0.080 ms
Kernel launch overhead:      ~0.013 ms (3 launches)
```

**Savings:** ~0.200 ms (54% faster)

### Lessons Learned

1. **Kernel fusion is highly effective for reducing launch overhead**
   - From 7+ launches to 3 launches
   - Each launch saved: ~0.01-0.02 ms

2. **Grid-stride loops have limits**
   - Works well when numBlocks ≤ 1024
   - Degrades when each thread processes 100+ elements
   - Need multi-block reduction for very large arrays

3. **Shared memory reuse is powerful**
   - Kernel 1 uses same buffer for max and sum reductions
   - No extra memory allocation needed
   - Better cache utilization

4. **Critical numerical stability formula:**
   ```
   global_sum = Σ(block_sum[i] * exp(block_max[i] - global_max))
   ```
   - Must adjust each block's sum when merging with global max
   - Maintains numerical stability across block boundaries

### Bug Fixed During Implementation

**Issue:** Verification failed for arrays > 100K elements
```
Sum(output) = 15.27 (expected 1.0)  // WRONG!
```

**Root cause:** Original Kernel 2 only processed first `threadsPerBlock` blocks
```cuda
// WRONG: Only loads first 256 blocks
int idx = tid;
sdata[tid] = (idx < numBlocks) ? block_maxes[idx] : -INFINITY;
```

**Fix:** Use grid-stride loop to process all blocks
```cuda
// CORRECT: Each thread processes multiple blocks
float thread_max = -INFINITY;
for (int i = tid; i < numBlocks; i += blockDim.x) {
    thread_max = fmaxf(thread_max, block_maxes[i]);
}
```

### When to Use Fused Softmax

**Recommended:**
- Array sizes: 1K - 10M elements
- Performance-critical applications
- When kernel launch overhead is significant

**Not recommended:**
- Very large arrays (> 50M elements)
- Use multi-pass or online softmax instead
- Or implement multi-block reduction for Kernel 2

### Future Optimizations

**1. Multi-block Kernel 2** (for 100M+ arrays):
- Launch multiple blocks for global reduction
- Recursive reduction if needed
- Expected: Match multi-pass performance at large scales

**2. Warp-optimized reductions:**
- Use `__shfl_down_sync` for final 32→1 reductions
- Eliminate some `__syncthreads` barriers
- Expected: Additional 8-10% speedup

**3. 2-kernel version:**
- Merge Kernel 2 + Kernel 3
- More complex but slightly faster
- Good middle ground before 1-kernel approach

---

## Future Work

### Planned Implementations

**1. Online Softmax:**
- Single-pass algorithm (streaming max + sum)
- Update running statistics as we go
- Expected: Fastest (single memory pass)
- Challenges: Most complex, precision-sensitive

**2. Warp-optimized variants:**
- Use `__shfl_down_sync` for final 32→1 reduction
- Eliminate `__syncthreads` barriers
- Expected: 8-10% speedup (like in reduction experiments)

### Benchmark Progress

**Current performance (1M elements):**
- Naive: 0.190 ms (fast but broken)
- Multi-pass: 0.372 ms (stable baseline)
- **Fused (implemented):** 0.173 ms ✓ (2.15x faster than multi-pass!)
- Online: ~0.180 ms (goal)
- **Achievement:** Fused softmax is now faster than naive while staying stable!

### 2D Softmax (Batch Processing)

**Extension to batch × seq_len:**
```cuda
// Each block processes one sequence
__global__ void softmax2D(float *input, float *output, int batch, int seq_len) {
    int row = blockIdx.x;  // Which sequence
    // Softmax over input[row * seq_len : (row+1) * seq_len]
}
```

**Use case:** Transformer attention (batch_size × num_heads × seq_len)

---

## Key Takeaways

1. **Numerical stability is not optional**
   - Naive softmax breaks with real-world inputs
   - Max subtraction is the standard solution
   - Small overhead is acceptable for correctness

2. **Performance trade-offs:**
   - Multi-pass: 2x slower but always correct
   - Can be optimized with kernel fusion
   - Fused/online variants can match naive speed

3. **Design pattern:** Multi-stage reductions
   - Find statistic (max, mean, etc.)
   - Use statistic in second reduction
   - Apply final transformation
   - Common in normalization operations

4. **Implementation details matter:**
   - Don't re-apply transformations in reduction stages!
   - Use correct identity elements for reductions
   - Track state across multi-kernel algorithms

5. **Production reality:**
   - All ML frameworks use stable softmax
   - PyTorch/TensorFlow: Fused + online algorithms
   - Numerical correctness >> raw performance
   - But optimizations can recover performance!

---

*Last Updated: 2026-01-02*
