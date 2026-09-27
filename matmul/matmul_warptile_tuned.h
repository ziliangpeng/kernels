#ifndef MATMUL_WARPTILE_TUNED_H
#define MATMUL_WARPTILE_TUNED_H

#include "matmul_kernel.h"

// Autotunable warp-tiling matmul (rung 10), dtype-templated.
// Config: (BM, BN, BK, TM, TN, WM, WN); warp layout fixed at 4x8 threads.

struct CfgW {
    int BM, BN, BK, TM, TN, WM, WN;
};

template <typename Traits>
class MatmulWarptileTuned : public MatmulKernel {
public:
    MatmulWarptileTuned(int N, int blockDim, const CfgW &c);
    ~MatmulWarptileTuned() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

    // Sweeps all compiled-in configs, prints CSV lines, installs the best.
    void autotune(const float *d_A, const float *d_B, int N, int num_iterations);

private:
    int N;
    int blockDim;
    CfgW cfg;
    typename Traits::T *d_A16 = nullptr;
    typename Traits::T *d_B16 = nullptr;
    bool converted = false;
};

#endif // MATMUL_WARPTILE_TUNED_H
