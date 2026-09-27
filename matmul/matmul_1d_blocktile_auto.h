#ifndef MATMUL_1D_BLOCKTILE_AUTO
#define MATMUL_1D_BLOCKTILE_AUTO

#include "matmul_kernel.h"

// Autotuning wrappers ported from archive-cudakernels (2026-09-26).
// Sweeps a hand-curated candidate grid on first execute(), keeps the best
// config; timing = 2 warmup + 3 timed, median. Originals: commits 4c01b74+
// in the archive repo; reproduced 19.2/34.1/34.9/39.3T on gcp5 H100.

class Matmul1DBlocktileAuto : public MatmulKernel {
private:
    int N;
    int blockDim;
    int best_BM;
    int best_BN;
    int best_BK;
    int best_TM;
    float best_time_ms;
    bool tuned;

    void tune(const float *d_A, const float *d_B, float *d_C);
    void launch(const float *d_A, const float *d_B, float *d_C,
                int BM, int BN, int BK, int TM);

public:
    Matmul1DBlocktileAuto(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~Matmul1DBlocktileAuto() override;
};

#endif // MATMUL_1D_BLOCKTILE_AUTO
