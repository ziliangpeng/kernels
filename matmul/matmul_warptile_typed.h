#ifndef MATMUL_WARPTILE_TYPED_H
#define MATMUL_WARPTILE_TYPED_H

#include "matmul_kernel.h"

// Warp-tiling matmul (rung 10), dtype-templated: three-level hierarchy
// Block -> Warp -> Thread, register reuse across the warp tile.
// FP16: 16-bit storage + FP32 accumulation (same semantics as the ladder).

template <typename Traits>
class MatmulWarptileTyped : public MatmulKernel {
public:
    MatmulWarptileTyped(int N, int blockDim);
    ~MatmulWarptileTyped() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#include "dtype_traits.h"

#endif // MATMUL_WARPTILE_TYPED_H
