#include "matmul_cublas.h"
#include "cuda_utils.h"
#include <iostream>

// Constructor: Create cuBLAS handle (setup phase, not timed)
MatmulCublas::MatmulCublas(int N, int blockDim) : N(N) {
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Execute: Pure kernel execution (this method is timed in benchmarks)
void MatmulCublas::execute(const float *d_A, const float *d_B, float *d_C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major ordering, but we have row-major matrices
    // To compute C = A * B in row-major, we compute C^T = B^T * A^T
    // which is equivalent to: C = B * A in column-major interpretation
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,           // M, N, K dimensions
                &alpha,
                d_B, N,            // Matrix B (interpreted as col-major)
                d_A, N,            // Matrix A (interpreted as col-major)
                &beta,
                d_C, N);           // Matrix C (output)

    cudaCheckError(cudaGetLastError());
}

// Destructor: Clean up cuBLAS handle
MatmulCublas::~MatmulCublas() {
    cublasDestroy(handle);
}
