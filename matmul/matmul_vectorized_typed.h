#ifndef MATMUL_VECTORIZED_TYPED_H
#define MATMUL_VECTORIZED_TYPED_H

#include "matmul_kernel.h"

// Vectorized GEMM (rung 6), dtype-templated: 16-byte vectorized GMEM loads
// (float4 for FP32 storage, half8/uint4 for FP16 storage), transposed As tile
// for coalesced SMEM reads, register reuse, vectorized C writes.
// FP16: 8 halves per 16-byte load — 8x fewer load instructions than scalar.

template <typename Traits>
class MatmulVectorizedTyped : public MatmulKernel {
public:
    MatmulVectorizedTyped(int N, int blockDim);
    ~MatmulVectorizedTyped() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#include "dtype_traits.h"

#endif // MATMUL_VECTORIZED_TYPED_H
