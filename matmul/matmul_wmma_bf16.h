#ifndef MATMUL_WMMA_BF16_H
#define MATMUL_WMMA_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// BF16 WMMA Tensor Core Matrix Multiplication
//
// Uses Tensor Cores with BF16 (bfloat16) precision.
// BF16 has the same exponent range as FP32 (8 bits) but reduced mantissa (7 bits).
// This makes it more numerically stable than FP16 for deep learning workloads.
//
// H100 BF16 Tensor Core peak: ~990 TFLOPS (2x FP32 Tensor Core)
//
// WMMA tile size: 16x16x16

class MatmulWmmaBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;  // BF16 buffer for A
    __nv_bfloat16 *d_B_bf16;  // BF16 buffer for B
    float *d_C_fp32;          // FP32 accumulator output

public:
    MatmulWmmaBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaBf16() override;
};

#endif
