#include "matmul_cublas_bf16.h"
#include "cuda_utils.h"
#include <cuda_bf16.h>
#include <iostream>

// FP32 to BF16 conversion kernel
__global__ void convertFP32ToBF16(const float* input, __nv_bfloat16* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

// Constructor
MatmulCublasBf16::MatmulCublasBf16(int N, int blockDim) : N(N) {
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate BF16 buffers for inputs
    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

// Execute
void MatmulCublasBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    // Convert FP32 inputs to BF16
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    convertFP32ToBF16<<<blocks, threads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16<<<blocks, threads>>>(d_B, d_B_bf16, N * N);
    cudaCheckError(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cublasGemmEx with BF16 inputs, FP32 output, FP32 compute
    // Note: cuBLAS uses column-major, so we compute C = B * A to get row-major result
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B_bf16, CUDA_R_16BF, N,  // B matrix (BF16)
        d_A_bf16, CUDA_R_16BF, N,  // A matrix (BF16)
        &beta,
        d_C, CUDA_R_32F, N,        // C matrix (FP32 output)
        CUBLAS_COMPUTE_32F,        // Compute in FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use Tensor Cores
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmEx failed with status: " << status << std::endl;
    }

    cudaCheckError(cudaGetLastError());
}

// Destructor
MatmulCublasBf16::~MatmulCublasBf16() {
    cublasDestroy(handle);
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
