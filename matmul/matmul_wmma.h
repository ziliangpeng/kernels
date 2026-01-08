#ifndef MATMUL_WMMA_H_
#define MATMUL_WMMA_H_

#include "matmul_kernel.h"
#include <cuda_fp16.h>

class MatmulWMMA : public MatmulKernel {
private:
    half *d_A_fp16;  // FP16 buffer for matrix A
    half *d_B_fp16;  // FP16 buffer for matrix B
    float *d_C_fp32; // FP32 buffer for accumulator
    int N;

public:
    MatmulWMMA(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWMMA() override;
};

#endif  // MATMUL_WMMA_H_
