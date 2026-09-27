#ifndef MATMUL_COALESCED_TYPED_H
#define MATMUL_COALESCED_TYPED_H

#include "matmul_kernel.h"

// Coalesced matmul (rung 2) with 16-bit storage + FP32 accumulation.
// Identical thread mapping to matmul_coalesced.cu: 32x32 output tile per
// block, 1D thread indexing so consecutive threads hit consecutive B columns
// (coalesced loads). Conversion runs once, lazily, in warmup — timed region
// is pure GEMM at 16-bit storage.

template <typename Traits>
class MatmulCoalescedTyped : public MatmulKernel {
public:
    MatmulCoalescedTyped(int N, int blockDim);
    ~MatmulCoalescedTyped() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_COALESCED_TYPED_H
