#ifndef MATMUL_VECTORIZED_TUNED_H
#define MATMUL_VECTORIZED_TUNED_H

#include "matmul_kernel.h"

// Autotunable vectorized matmul (rung 6), dtype-templated.
// Config: (BM, BN, BK, TM, TN); 16-byte vector loads (float4 / half8);
// threads = (BM/TM)*(BN/TN) <= 1024.

struct CfgVec {
    int BM, BN, BK, TM, TN;
};

template <typename Traits>
class MatmulVectorizedTuned : public MatmulKernel {
public:
    MatmulVectorizedTuned(int N, int blockDim, const CfgVec &c);
    ~MatmulVectorizedTuned() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

    // Sweeps all compiled-in configs, prints CSV lines (idx,BM,BN,BK,TM,TN,GFLOPS),
    // installs the best into cfg.
    void autotune(const float *d_A, const float *d_B, int N, int num_iterations);

private:
    int N;
    int blockDim;
    CfgVec cfg;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_VECTORIZED_TUNED_H
