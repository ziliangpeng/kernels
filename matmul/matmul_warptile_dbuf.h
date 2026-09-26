#ifndef MATMUL_WARPTILE_DBUF_H
#define MATMUL_WARPTILE_DBUF_H

#include "matmul_kernel.h"

// Warp Tiling + cp.async double buffering (Simon rung 12)
// Adds async GMEM->SMEM copies (cuda::memcpy_async + cuda::barrier) on top of
// the warptile kernel, overlapping the next tile's load with the current
// tile's compute.

class MatmulWarptileDbuf : public MatmulKernel {
public:
    MatmulWarptileDbuf(int N, int blockDim);
    ~MatmulWarptileDbuf() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
};

#endif // MATMUL_WARPTILE_DBUF_H
