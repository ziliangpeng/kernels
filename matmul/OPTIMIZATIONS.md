# Matrix Multiplication Optimization Roadmap

This document outlines all possible optimizations and kernel variants for the matmul project, organized from basic to advanced techniques.

## 🎯 Overview

Matrix multiplication is one of the most heavily optimized operations in computing. This roadmap progresses from simple memory optimizations (10-20x speedup) to advanced Tensor Core implementations (1000x+ speedup).

**Current Status**: ✅ Naive kernel implemented (~5 GFLOPS baseline)

---

## 📊 Optimization Tiers

### **Tier 1: Memory Hierarchy Optimizations** ⭐ START HERE

These optimizations focus on efficient use of the GPU memory hierarchy and provide the biggest bang-for-buck improvements.

#### 1. **Tiled/Shared Memory Matmul** (10-20x speedup)
**Priority**: HIGHEST - This is THE fundamental GPU matmul optimization

**Concept**:
- Load tiles of A and B into shared memory
- Reuse tiles across multiple output elements
- Reduces global memory accesses from O(N³) to O(N³/TILE_SIZE)

**Implementation Details**:
```cuda
// Each block computes TILE_SIZE × TILE_SIZE output elements
// Load tiles cooperatively into shared memory
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

for (int t = 0; t < N / TILE_SIZE; t++) {
    // Load tile collaboratively
    As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
    Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
    __syncthreads();

    // Compute using shared memory
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
}
```

**Files to create**:
- `matmul_tiled_basic.cu` - Simple 16×16 or 32×32 tiling
- `matmul_tiled_optimized.cu` - With memory coalescing optimizations

**Recommended tile sizes**: 16×16, 32×32 (experiment to find optimal)

---

#### 2. **Register Tiling** (2-3x over basic tiling)
**Priority**: HIGH - Complements shared memory tiling

**Concept**:
- Each thread computes multiple output elements (e.g., 4×4 or 8×8 block)
- Keep partial results in registers (fastest memory)
- Use outer product formulation: C[i:i+R][j:j+R] += A[i:i+R][k] * B[k][j:j+R]

**Implementation Details**:
```cuda
// Each thread computes THREAD_TILE_Y × THREAD_TILE_X outputs
float regC[THREAD_TILE_Y][THREAD_TILE_X] = {0};
float regA[THREAD_TILE_Y];
float regB[THREAD_TILE_X];

for (int k = 0; k < TILE_SIZE; k++) {
    // Load into registers
    for (int i = 0; i < THREAD_TILE_Y; i++)
        regA[i] = As[ty * THREAD_TILE_Y + i][k];
    for (int j = 0; j < THREAD_TILE_X; j++)
        regB[j] = Bs[k][tx * THREAD_TILE_X + j];

    // Outer product
    for (int i = 0; i < THREAD_TILE_Y; i++)
        for (int j = 0; j < THREAD_TILE_X; j++)
            regC[i][j] += regA[i] * regB[j];
}
```

**Files to create**:
- `matmul_register_tiled.cu` - 4×4 or 8×8 thread tiles

**Recommended thread tile sizes**: 4×4, 8×8, 4×8

---

#### 3. **Double Buffering** (1.5-2x over basic tiling)
**Priority**: MEDIUM - Overlaps computation with memory loads

**Concept**:
- Use two sets of shared memory buffers
- While computing on buffer A, prefetch next tile into buffer B
- Hides memory latency behind computation

**Implementation Details**:
```cuda
__shared__ float As[2][TILE_SIZE][TILE_SIZE];  // Double buffer
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

int write_idx = 0, read_idx = 1;

// Prefetch first tile
load_tile(As[write_idx], Bs[write_idx], 0);
__syncthreads();

for (int t = 1; t < num_tiles; t++) {
    // Swap buffers
    read_idx = write_idx;
    write_idx = 1 - write_idx;

    // Load next tile (overlapped with compute)
    load_tile(As[write_idx], Bs[write_idx], t);

    // Compute current tile
    compute_tile(As[read_idx], Bs[read_idx]);

    __syncthreads();
}
```

**Files to create**:
- `matmul_double_buffer.cu`

---

### **Tier 2: Warp-Level Optimizations**

Leverage warp-level primitives for intra-warp communication.

#### 4. **Warp Shuffle Matmul** (1.5-2x over shared memory)
**Priority**: MEDIUM

**Concept**:
- Use `__shfl_sync()` for broadcasting values within a warp
- Faster than shared memory (no synchronization overhead)
- Reduces shared memory pressure

**Implementation Details**:
```cuda
// Distribute row of A across warp lanes
float a_val = A[row * N + lane_id];

for (int k = 0; k < N; k++) {
    // Each lane loads one element of B column
    float b_val = B[k * N + col];

    // Get corresponding A value via shuffle
    float a_broadcast = __shfl_sync(0xffffffff, a_val, k % 32);

    sum += a_broadcast * b_val;
}
```

**Files to create**:
- `matmul_warp_shuffle.cu`

---

#### 5. **Size-Specialized Kernels**
**Priority**: MEDIUM - Optimize for specific matrix sizes

**Small matrices (<128×128)**:
- `matmul_tiny.cu` - Single warp handles entire matrix
- Similar to `softmax_tiny` pattern

**Medium matrices (128-1024)**:
- `matmul_small.cu` - Single block with hybrid reduction
- Similar to `softmax_small` pattern

**Large matrices (>1024)**:
- Use tiled kernels with optimal configurations

---

### **Tier 3: Vectorization & Memory Access Patterns**

Optimize memory bandwidth utilization.

#### 6. **Vectorized Loads (float4)** (1.5-2x speedup)
**Priority**: MEDIUM

**Concept**:
- Load 4 consecutive floats in a single transaction
- Improves memory bandwidth utilization
- Works best when accessing contiguous memory

**Implementation Details**:
```cuda
// Load 4 elements at once
float4 a4 = *reinterpret_cast<const float4*>(&A[offset]);
float a_vals[4] = {a4.x, a4.y, a4.z, a4.w};

// Process 4 elements per iteration
for (int k = 0; k < N; k += 4) {
    float4 b4 = *reinterpret_cast<const float4*>(&B[k * N + col]);
    sum += a_vals[0] * b4.x + a_vals[1] * b4.y +
           a_vals[2] * b4.z + a_vals[3] * b4.w;
}
```

**Files to create**:
- `matmul_vectorized.cu`

**Note**: Requires matrix dimensions to be multiples of 4

---

#### 7. **Bank Conflict Elimination** (1.2-1.5x speedup)
**Priority**: LOW-MEDIUM

**Concept**:
- Shared memory is organized into 32 banks
- Simultaneous access to same bank causes serialization
- Pad shared memory to avoid conflicts

**Implementation Details**:
```cuda
// Without padding (potential bank conflicts)
__shared__ float As[TILE_SIZE][TILE_SIZE];

// With padding (avoids bank conflicts)
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 column padding

// Access pattern remains the same
As[ty][tx] = A[...];
```

**Files to create**:
- `matmul_no_bank_conflict.cu`

---

#### 8. **Async Memory Copy & Prefetching** (1.2-1.5x speedup)
**Priority**: MEDIUM (requires Ampere+ GPU)

**Concept**:
- Use `cp.async` instruction for asynchronous global-to-shared memory copy
- Prefetch next tile while computing current one

**Implementation Details**:
```cuda
// Requires CUDA 11.0+, Compute Capability 8.0+
__pipeline_memcpy_async(&As[write][ty][tx],
                        &A[global_offset],
                        sizeof(float));
__pipeline_commit();

// Compute while memory copy is in flight
compute_tile();

__pipeline_wait_prior(0);
__syncthreads();
```

**Files to create**:
- `matmul_async_copy.cu`
- `matmul_prefetch.cu`

---

### **Tier 4: Advanced Instruction-Level Parallelism**

Leverage specialized hardware instructions for maximum performance.

#### 9. **Tensor Core Matmul** ⭐ HIGHEST PERFORMANCE (10-100x speedup)
**Priority**: HIGHEST (for modern GPUs)

**Hardware Requirements**:
- Volta (SM 7.0): FP16 Tensor Cores
- Turing (SM 7.5): FP16, INT8, INT4
- Ampere (SM 8.0+): TF32, BF16, FP64
- Hopper (SM 9.0): FP8

**Variants to implement**:

##### 9a. **WMMA (Warp Matrix Multiply-Accumulate)** - High-level API
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// FP16: 16×16×16 matrix multiply
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

load_matrix_sync(a_frag, A, 16);
load_matrix_sync(b_frag, B, 16);
fill_fragment(c_frag, 0.0f);

mma_sync(c_frag, a_frag, b_frag, c_frag);

store_matrix_sync(C, c_frag, 16, mem_row_major);
```

**Files to create**:
- `matmul_wmma_fp16.cu` - FP16 input, FP32 accumulate
- `matmul_wmma_tf32.cu` - TF32 mode (Ampere+)
- `matmul_wmma_int8.cu` - INT8 for inference

##### 9b. **Direct PTX (MMA instruction)** - Low-level control
```cuda
// Direct PTX for maximum control
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
);
```

**Files to create**:
- `matmul_mma_ptx.cu` - Direct PTX for experts

##### 9c. **CUTLASS Integration** - Template library
```cpp
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                          // Element A
    cutlass::layout::RowMajor,               // Layout A
    cutlass::half_t,                          // Element B
    cutlass::layout::ColumnMajor,            // Layout B
    float,                                    // Element C
    cutlass::layout::RowMajor,               // Layout C
    float,                                    // Element Accumulator
    cutlass::arch::OpClassTensorOp,          // Operator class (Tensor Core)
    cutlass::arch::Sm80                      // Architecture (Ampere)
>;

Gemm gemm_op;
gemm_op({{M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1.0f, 0.0f}});
```

**Files to create**:
- `matmul_cutlass.cu` - Using CUTLASS library

**Performance expectations**:
- WMMA: 2-10 TFLOPS (depending on GPU)
- Well-optimized Tensor Core: 10-50 TFLOPS
- cuBLAS (uses Tensor Cores): 10-100+ TFLOPS

---

#### 10. **Mixed Precision Variants**
**Priority**: MEDIUM (for specific use cases)

**Variants**:
- `matmul_fp16_fp32.cu` - FP16 multiply, FP32 accumulate (ML workloads)
- `matmul_int8.cu` - INT8 quantized (inference)
- `matmul_tf32.cu` - TensorFloat-32 (Ampere+, good accuracy/speed tradeoff)
- `matmul_bf16.cu` - BFloat16 (ML training)

---

### **Tier 5: Algorithmic Variants**

Different algorithms with different complexity characteristics.

#### 11. **Strassen Algorithm** (1.3-1.5x for large matrices)
**Priority**: LOW - Only beneficial for very large matrices (>2048)

**Concept**:
- Reduce complexity from O(N³) to O(N^2.807)
- Uses 7 multiplications instead of 8 via clever matrix decomposition
- Recursive algorithm

**Implementation**:
- Break matrices into 4 quadrants
- Compute 7 products using Strassen formulas
- Recursively apply until reaching base case (use optimized kernel)

**Files to create**:
- `matmul_strassen.cu`

**Challenges**:
- More memory allocations
- Numerical stability concerns
- Only faster for very large matrices

---

#### 12. **Block-Recursive Matmul**
**Priority**: LOW - Academic interest

**Concept**:
- Cache-oblivious algorithm
- Recursively divide matrix until it fits in cache

**Files to create**:
- `matmul_recursive.cu`

---

#### 13. **Batched Matmul** (Important for ML)
**Priority**: HIGH (for batch workloads)

**Concept**:
- Compute many small matmuls in parallel
- Common in deep learning (batch of inputs)

**Variants**:
- Fixed-size batched matmul
- Variable-size batched matmul (more complex)

**Files to create**:
- `matmul_batched_fixed.cu` - All matrices same size
- `matmul_batched_variable.cu` - Different sizes

**cuBLAS alternative**: `cublasSgemmBatched()`

---

### **Tier 6: Library Baselines** ⭐ ESSENTIAL

These are production-quality implementations for comparison.

#### 14. **cuBLAS** - NVIDIA's optimized BLAS library
**Priority**: HIGHEST - Essential baseline for comparison

**Concept**:
- Highly optimized by NVIDIA engineers
- Uses Tensor Cores when available
- Production-quality, industry standard

**Implementation**:
```cpp
cublasHandle_t handle;
cublasCreate(&handle);

const float alpha = 1.0f;
const float beta = 0.0f;

cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N);

cublasDestroy(handle);
```

**Files to create**:
- `matmul_cublas.cu` - Standard cuBLAS
- `matmul_cublas_tensor.cu` - Explicitly use Tensor Cores (cublasGemmEx)

**Expected performance**: 5-100+ TFLOPS depending on GPU

---

#### 15. **CUTLASS** - NVIDIA's template library
**Priority**: HIGH

**Concept**:
- Template-based GEMM implementations
- Exposes fine-grained control over Tensor Core operations
- Modern C++ design

**Files to create**:
- `matmul_cutlass_basic.cu` - Basic CUTLASS usage
- `matmul_cutlass_tuned.cu` - Custom tile sizes and thread layouts

---

#### 16. **CuDNN (if applicable)**
**Priority**: MEDIUM

**Concept**:
- Optimized for deep learning workloads
- May have specialized convolution implementations

**Files to create**:
- `matmul_cudnn.cu`

---

## 📈 Expected Performance Ladder

Based on typical A100 GPU performance:

| Kernel                    | GFLOPS    | Speedup | Priority      |
|---------------------------|-----------|---------|---------------|
| ✅ Naive                  | ~5        | 1x      | Done          |
| Tiled (shared memory)     | ~100      | 20x     | ⭐ Start here |
| Register tiled            | ~300      | 60x     | High          |
| + Vectorized              | ~500      | 100x    | Medium        |
| + Double buffer           | ~800      | 160x    | Medium        |
| WMMA (Tensor Cores)       | ~5,000    | 1,000x  | ⭐ High perf  |
| CUTLASS (optimized)       | ~10,000   | 2,000x  | High          |
| cuBLAS (best)             | ~15,000+  | 3,000x+ | ⭐ Baseline   |

*Note: Numbers are rough estimates and vary by matrix size, GPU model, and implementation quality.*

---

## 🚀 Recommended Implementation Order

### **Phase 1: Foundation (Week 1)**
Priority: Build the essential optimizations and baseline.

1. ✅ **Naive** (Done) - Baseline
2. ⭐ **Tiled/Shared Memory** - THE fundamental optimization
3. ⭐ **cuBLAS** - The baseline to beat

**Goal**: Understand the 20x improvement from tiling and establish production baseline.

---

### **Phase 2: Refinements (Week 2)**
Priority: Stack multiple optimizations for cumulative improvements.

4. **Register Tiling** - 2-3x improvement
5. **Vectorized Loads** - 1.5-2x improvement
6. **Bank Conflict Elimination** - 1.2-1.5x improvement

**Goal**: Reach 100-500 GFLOPS through careful optimization.

---

### **Phase 3: Advanced Hardware (Week 3)**
Priority: Leverage specialized instructions for maximum performance.

7. ⭐ **WMMA Tensor Cores** - 10-100x improvement (if available)
8. **Double Buffering** - Overlap memory and compute
9. **Warp Shuffle** - Reduce shared memory usage

**Goal**: Approach or exceed 1 TFLOPS using Tensor Cores.

---

### **Phase 4: Specialized & Production (Week 4)**
Priority: Handle special cases and production workloads.

10. **Size-specialized kernels** (tiny/small/large)
11. **Batched operations** (for ML workloads)
12. **CUTLASS integration** (template library)
13. **Mixed precision variants** (FP16, INT8)

**Goal**: Production-ready kernels for various use cases.

---

## 🎓 Learning Resources

### **Books**
- "Programming Massively Parallel Processors" by Hwu, Kirk, and Hajj
  - Chapter 4: Memory Architecture
  - Chapter 9: Advanced Patterns (tiling)

### **NVIDIA Documentation**
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  - Shared Memory
  - Warp-level primitives
  - Tensor Cores (WMMA)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  - Memory Optimization
  - Instruction Optimization
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)

### **Papers**
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn, 2008)
- "Fast and Memory-Efficient Gemm" (CUTLASS papers)
- "Optimization of Matrix Multiplication on GPUs" (Jia et al.)

### **Code Examples**
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - Production-quality templates
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) - matrixMul example
- [Simon's Blog](https://siboehm.com/articles/22/CUDA-MMM) - Excellent tutorial

---

## 💡 Implementation Guidelines

### **Adding a New Kernel**

1. **Create the kernel files**:
   ```bash
   cuda/matmul/matmul_<name>.h
   cuda/matmul/matmul_<name>.cu
   ```

2. **Follow the class pattern**:
   ```cpp
   class Matmul<Name> : public MatmulKernel {
   private:
       int N, blockDim;
       // Any workspace needed

   public:
       Matmul<Name>(int N, int blockDim);
       void execute(const float *d_A, const float *d_B, float *d_C) override;
       ~Matmul<Name>() override;
   };
   ```

3. **Update BUILD file**:
   ```python
   cuda_library(
       name = "matmul_<name>",
       srcs = ["matmul_<name>.cu"],
       hdrs = ["matmul_<name>.h", "matmul_kernel.h"],
       deps = ["//cuda:cuda_utils"],
       ...
   )
   ```

4. **Update matmul.cpp**:
   ```cpp
   // Add to includes
   #include "matmul_<name>.h"

   // Add to BENCHMARK_METHODS array
   const char* BENCHMARK_METHODS[] = {
       "naive",
       "<name>",  // Add here
   };

   // Add to benchmark instantiation
   if (strcmp(method, "<name>") == 0) {
       kernel = new Matmul<Name>(N, blockDim);
   }
   ```

5. **Test**:
   ```bash
   bazel build //cuda/matmul:matmul --platforms=//platforms:cuda
   ./bazel-bin/cuda/matmul/matmul --method <name> -n 256 --verify
   ./bazel-bin/cuda/matmul/matmul --method all --verify
   ```

---

### **Testing Methodology**

For each new kernel:

1. **Correctness**: Always run with `--verify` first
2. **Performance**: Benchmark across all sizes
3. **Profiling**: Use `nsys` or `ncu` for detailed analysis
   ```bash
   nsys profile ./matmul --method <name> -n 1024
   ncu --metrics all ./matmul --method <name> -n 1024
   ```
4. **Comparison**: Compare against cuBLAS and previous kernels

---

### **Documentation Standards**

Each kernel should have:
- **Header comment**: Explain the optimization technique
- **Performance characteristics**: Expected speedup
- **Hardware requirements**: Compute capability, memory
- **Limitations**: Matrix size constraints, alignment requirements
- **References**: Papers or resources

Example:
```cpp
// Tiled matrix multiplication using shared memory
//
// OPTIMIZATION: Reduces global memory accesses by loading tiles into shared
// memory and reusing them across multiple output computations.
//
// PERFORMANCE: ~20x speedup over naive implementation
//
// REQUIREMENTS:
// - N must be divisible by TILE_SIZE
// - TILE_SIZE typically 16 or 32 (balance occupancy vs shared memory)
//
// REFERENCE: "Programming Massively Parallel Processors" Chapter 4
```

---

## 🎯 Success Metrics

Track these metrics for each kernel:

1. **Execution Time** (ms)
2. **GFLOPS** = (2×N³ - N²) / (time × 10⁶)
3. **Speedup** vs naive
4. **Efficiency** = GFLOPS / Theoretical Peak
5. **Memory Bandwidth Utilization**
6. **Occupancy**

Include these in benchmark output or a summary table.

---

## 🔬 Advanced Topics (Future)

Once basics are solid, explore:

1. **Auto-tuning**: Automatically find optimal tile sizes
2. **Multi-GPU**: Distribute computation across GPUs
3. **Sparse Matmul**: Optimize for sparse matrices
4. **Quantization**: INT8, INT4 for inference
5. **Custom Data Layouts**: Block sparse, CSR, etc.
6. **JIT Compilation**: Runtime code generation for specific sizes
7. **Mixed-Precision Training**: FP16/BF16 forward, FP32 backward

---

## 📊 Profiling and Analysis

### **Tools**
- **nsys** (Nsight Systems): System-wide timeline
- **ncu** (Nsight Compute): Kernel-level metrics
- **nvprof** (legacy): Simple profiling

### **Key Metrics to Track**
- Achieved Occupancy
- Memory Throughput (% of peak bandwidth)
- Compute Throughput (% of peak FLOPS)
- Warp Execution Efficiency
- Shared Memory Bank Conflicts
- L1/L2 Cache Hit Rates

### **Example Profiling Session**
```bash
# System-wide timeline
nsys profile -o matmul_tiled ./matmul --method tiled -n 1024

# Detailed kernel metrics
ncu --set full -o matmul_tiled ./matmul --method tiled -n 1024

# Compare two kernels
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./matmul --method naive -n 1024
ncu --set full --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./matmul --method tiled -n 1024
```

---

## 🎉 Summary

This roadmap provides a structured path from basic (naive) to cutting-edge (Tensor Core) matrix multiplication implementations. Start with tiled/shared memory for the biggest improvement, establish cuBLAS as your baseline, then progressively add optimizations while measuring performance gains at each step.

**Remember**:
- Each optimization builds on the previous ones
- Always verify correctness before benchmarking performance
- Profile to understand bottlenecks before optimizing
- Compare against cuBLAS to know how far you've come

Good luck optimizing! 🚀
