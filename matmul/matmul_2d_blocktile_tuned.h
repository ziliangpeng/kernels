#ifndef MATMUL_2D_BLOCKTILE_TUNED_H
#define MATMUL_2D_BLOCKTILE_TUNED_H

#include "matmul_kernel.h"

// Autotunable 2D block-tiling matmul (rung 5), dtype-templated.
// Config: (BM, BN, BK, TM, TN) with threads = (BM/TM)*(BN/TN) <= 1024.

struct Cfg2D {
    int BM, BN, BK, TM, TN;
};

template <typename Traits>
class Matmul2DBlocktileTuned : public MatmulKernel {
public:
    Matmul2DBlocktileTuned(int N, int blockDim, const Cfg2D &c);
    ~Matmul2DBlocktileTuned() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

    // Sweeps all compiled-in configs, prints CSV lines (idx,BM,BN,BK,TM,TN,GFLOPS),
    // installs the best into cfg.
    void autotune(const float *d_A, const float *d_B, int N, int num_iterations);

private:
    int N;
    int blockDim;
    Cfg2D cfg;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_2D_BLOCKTILE_TUNED_H
