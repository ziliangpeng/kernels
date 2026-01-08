#ifndef MATMUL_CUBLAS_H_
#define MATMUL_CUBLAS_H_

#include "matmul_kernel.h"
#include <cublas_v2.h>

class MatmulCublas : public MatmulKernel {
private:
    cublasHandle_t handle;  // cuBLAS context
    int N;                  // Matrix dimension

public:
    MatmulCublas(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulCublas() override;
};

#endif  // MATMUL_CUBLAS_H_
