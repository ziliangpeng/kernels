#ifndef MATMUL_WARPTILE_DBUF_F16_H
#define MATMUL_WARPTILE_DBUF_F16_H

#include "matmul_kernel.h"
#include "dtype_traits.h"

// Warp Tiling + cp.async double buffering, FP16 storage / FP32 accumulation
// (16-bit ladder, rung 12). FP16 port of matmul_warptile_dbuf.cu with two
// layout changes required by cp.async's 4/8/16B granularity:
//   - A tile: NATURAL layout As[2][BM][BK+2] (FP32 version used transposed
//     As[2][BK][BM]; a half is 2B so transposed 4B chunks can't be coalesced
//     along k). Padded by +2 halves so a k-row stride is BK+2 halves = odd
//     number of 4B words when BK is even -> gcd(BK/2+1, 32)=1 -> no bank
//     conflicts on k-column walks.
//   - B tile: Bs[2][BK][BN], moved with 16B = 8-half chunks along N
//     (same vectorization factor as the FP32 version's float4).

template <typename Traits>
class MatmulWarptileDbufF16 : public MatmulKernel {
public:
    MatmulWarptileDbufF16(int N, int blockDim,
                          int BM = 128, int BN = 256, int BK = 8,
                          int WM = 64, int WN = 64, int WNITER = 2,
                          int TM = 8, int TN = 4, int NUM_THREADS = 256);
    ~MatmulWarptileDbufF16() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    int blockDim;
    typename Traits::T *d_A16;
    typename Traits::T *d_B16;
    int cfg[10];  // BM,BN,BK,WM,WN,WNITER,TM,TN,NT (9 used)
};

#endif // MATMUL_WARPTILE_DBUF_F16_H
