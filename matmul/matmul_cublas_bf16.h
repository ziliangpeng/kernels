#ifndef MATMUL_CUBLAS_BF16_H
#define MATMUL_CUBLAS_BF16_H

#include "matmul_kernel.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>

// cuBLAS BF16 Matrix Multiplication
//
// Uses cublasGemmEx with BF16 inputs and FP32 accumulation.
// This is the standard configuration for LLM training (BF16 compute, FP32 accum).
//
// H100 BF16 Tensor Core peak: ~990 TFLOPS

class MatmulCublasBf16 : public MatmulKernel {
private:
    int N;
    cublasHandle_t handle;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulCublasBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulCublasBf16() override;
};

#endif
