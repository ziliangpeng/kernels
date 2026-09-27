#ifndef MATMUL_1D_BLOCKTILE_TYPED_H
#define MATMUL_1D_BLOCKTILE_TYPED_H

#include "matmul_kernel.h"

// 1D block-tiling matmul (rung 4, NOT autotuned — default config
// BM=64 BN=64 BK=8 TM=8) with FP16 storage + FP32 accumulation.
// Same structure as matmul_1d_blocktile.cu: each thread computes TM=8
// outputs in a column, register reuse of B across TM results, SMEM tiles
// of A and B. A/B tiles staged in SMEM as FP16; compute loop converts
// per SMEM read (hot-path cvt, same semantics as smem_f16).

template <typename Traits>
class Matmul1DBlocktileTyped : public MatmulKernel {
public:
    Matmul1DBlocktileTyped(int N, int blockDim);
    ~Matmul1DBlocktileTyped() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_1D_BLOCKTILE_TYPED_H
