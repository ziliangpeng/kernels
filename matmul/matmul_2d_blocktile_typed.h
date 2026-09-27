#ifndef MATMUL_2D_BLOCKTILE_TYPED_H
#define MATMUL_2D_BLOCKTILE_TYPED_H

#include "matmul_kernel.h"

// 2D block-tiling matmul (rung 5, default config BM=64 BN=64 BK=16 TM=TN=8)
// with FP16 storage + FP32 accumulation.
//
// Hypothesis under test: the 2D outer-product structure amortizes the FP16
// cvt tax far better than 1D — each As/Bs element is cvt'd ONCE per dotIdx
// into regA/regB and reused TM*TN (=64) times in the outer-product FMA nest.
// Expected FP16/FP32 ratio >= 1.0x (vs 0.98x at 1D default).

template <typename Traits>
class Matmul2DBlocktileTyped : public MatmulKernel {
public:
    Matmul2DBlocktileTyped(int N, int blockDim);
    ~Matmul2DBlocktileTyped() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_2D_BLOCKTILE_TYPED_H
