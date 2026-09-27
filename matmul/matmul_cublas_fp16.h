#ifndef MATMUL_CUBLAS_FP16_H
#define MATMUL_CUBLAS_FP16_H

#include "matmul_kernel.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

// cuBLAS FP16 Tensor Core reference: FP16 inputs, FP32 output, FP32 compute
// (CUBLAS_COMPUTE_32F) — the fair ceiling for our FP16-storage + FP32-accum
// hand kernels. Conversion happens ONCE in the constructor (not per execute),
// so the timed region is pure cublasGemmEx.
class MatmulCublasFp16 : public MatmulKernel {
public:
    MatmulCublasFp16(int N, int blockDim);
    ~MatmulCublasFp16() override;

    void execute(const float *d_A, const float *d_B, float *d_C) override;

private:
    int N;
    cublasHandle_t handle;
    __half *d_A_h = nullptr;
    __half *d_B_h = nullptr;
};

#endif // MATMUL_CUBLAS_FP16_H
