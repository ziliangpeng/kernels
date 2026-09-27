#ifndef MATMUL_1D_BLOCKTILE_TUNED_H
#define MATMUL_1D_BLOCKTILE_TUNED_H

#include "matmul_kernel.h"

// Autotunable 1D block-tiling matmul (rung 4), dtype-templated.
// Config: (BM, BN, BK, TM) with the original 1D constraint BM = BN = BK*TM,
// threads = BN * BM / TM <= 1024. FP32 and FP16 share one code path.

struct Cfg1D {
    int BM, BN, BK, TM;
};

template <typename Traits>
class Matmul1DBlocktileTuned : public MatmulKernel {
public:
    Matmul1DBlocktileTuned(int N, int blockDim, const Cfg1D &c);
    ~Matmul1DBlocktileTuned() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

    // Sweeps all compiled-in configs, prints CSV lines (idx,BM,BN,BK,TM,GFLOPS),
    // installs the best into cfg.
    void autotune(const float *d_A, const float *d_B, int N, int num_iterations);

private:
    int N;
    int blockDim;
    Cfg1D cfg;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_1D_BLOCKTILE_TUNED_H
