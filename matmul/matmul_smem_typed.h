#ifndef MATMUL_SMEM_TYPED_H
#define MATMUL_SMEM_TYPED_H

#include "matmul_kernel.h"

// SMEM tiling matmul (rung 3) with 16-bit storage + FP32 accumulation.
// Same structure as matmul_smem.cu: 32x32 SMEM tiles of A and B, one thread
// per output element, K-loop over tiles with double __syncthreads.
// A/B tiles live in shared memory in the 16-bit format; the per-thread
// partial dot product converts each element to float on load from SMEM.
// GMEM→SMEM tile traffic halves vs FP32 (this rung's hypothesis: first
// 16-bit crossover point).

template <typename Traits>
class MatmulSmemTyped : public MatmulKernel {
public:
    MatmulSmemTyped(int N, int blockDim);
    ~MatmulSmemTyped() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_SMEM_TYPED_H
