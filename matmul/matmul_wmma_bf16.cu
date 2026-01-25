#include "matmul_wmma_bf16.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

// FP32 to BF16 conversion kernel
__global__ void convertFP32ToBF16Kernel(const float* input, __nv_bfloat16* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

// BF16 WMMA matmul kernel - same structure as FP16 WMMA
__global__ void matmulWmmaBf16Kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int N) {
#if __CUDA_ARCH__ >= 800
    // WMMA tile dimensions
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Calculate warp position (same as FP16 WMMA)
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    // Bounds check
    if (warpM * WMMA_M >= N || warpN * WMMA_N >= N) return;

    // Declare WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // Compute C = A × B by iterating over K dimension
    for (int k = 0; k < N; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Load fragments from A and B (both row-major)
        nvcuda::wmma::load_matrix_sync(a_frag, A + aRow * N + aCol, N);
        nvcuda::wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

        // Perform matrix multiply-accumulate
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, nvcuda::wmma::mem_row_major);
#endif
}

// Constructor
MatmulWmmaBf16::MatmulWmmaBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("BF16 WMMA requires compute capability 8.0+ (Ampere or newer)");
    }

    if (N % 16 != 0) {
        throw std::runtime_error("WMMA kernel requires matrix dimension N to be a multiple of 16");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_C_fp32, N * N * sizeof(float)));
}

// Execute
void MatmulWmmaBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    // Convert FP32 to BF16
    int convThreads = 256;
    int convBlocks = (N * N + convThreads - 1) / convThreads;
    convertFP32ToBF16Kernel<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Kernel<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);
    cudaCheckError(cudaGetLastError());

    // Zero output buffer
    cudaCheckError(cudaMemset(d_C_fp32, 0, N * N * sizeof(float)));

    // Configure WMMA kernel (same as FP16 WMMA)
    // Each warp handles one 16×16 output tile
    // Each block has 16 warps arranged as 4×4, handling a 64×64 output region
    dim3 blockDim(128, 4);  // 512 threads = 16 warps per block
    dim3 gridDim((N + 63) / 64, (N + 63) / 64);  // Grid based on 64×64 tiles per block

    matmulWmmaBf16Kernel<<<gridDim, blockDim>>>(d_A_bf16, d_B_bf16, d_C_fp32, N);
    cudaCheckError(cudaGetLastError());

    // Copy result
    cudaCheckError(cudaMemcpy(d_C, d_C_fp32, N * N * sizeof(float), cudaMemcpyDeviceToDevice));
}

// Destructor
MatmulWmmaBf16::~MatmulWmmaBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
    cudaFree(d_C_fp32);
}
