#include "matmul_wmma.h"
#include "cuda_utils.h"
#include <mma.h>
#include <iostream>
#include <stdexcept>

// FP32 to FP16 conversion kernel
__global__ void convertFP32ToFP16(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// WMMA matmul kernel using Tensor Cores
__global__ void matmulWMMAKernel(const half* A, const half* B, float* C, int N) {
    // WMMA tile dimensions
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Calculate warp position
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    // Bounds check
    if (warpM * WMMA_M >= N || warpN * WMMA_N >= N) return;

    // Declare WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
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
}

// Constructor: Allocate FP16 buffers
MatmulWMMA::MatmulWMMA(int N, int blockDim) : N(N) {
    // Check compute capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 7) {
        std::cerr << "WMMA requires compute capability 7.0+ (Volta or newer)" << std::endl;
        std::cerr << "Current GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
        throw std::runtime_error("WMMA requires compute capability 7.0+ (Volta or newer)");
    }

    // Check matrix dimension compatibility
    if (N % 16 != 0) {
        std::cerr << "WMMA kernel requires matrix dimension N to be a multiple of 16" << std::endl;
        std::cerr << "Current N: " << N << std::endl;
        throw std::runtime_error("WMMA kernel requires matrix dimension N to be a multiple of 16");
    }

    // Allocate FP16 buffers for input matrices
    cudaCheckError(cudaMalloc(&d_A_fp16, N * N * sizeof(half)));
    cudaCheckError(cudaMalloc(&d_B_fp16, N * N * sizeof(half)));
    cudaCheckError(cudaMalloc(&d_C_fp32, N * N * sizeof(float)));
}

// Execute: Convert to FP16, run WMMA, copy result
void MatmulWMMA::execute(const float *d_A, const float *d_B, float *d_C) {
    // Convert FP32 inputs to FP16
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    convertFP32ToFP16<<<blocks, threads>>>(d_A, d_A_fp16, N * N);
    convertFP32ToFP16<<<blocks, threads>>>(d_B, d_B_fp16, N * N);
    cudaCheckError(cudaGetLastError());

    // Zero output buffer to avoid garbage in uncomputed tiles
    cudaCheckError(cudaMemset(d_C_fp32, 0, N * N * sizeof(float)));

    // Configure WMMA kernel
    // Each warp handles one 16×16 output tile
    // Each block has 16 warps arranged as 4×4, handling a 64×64 output region
    dim3 blockDim(128, 4);  // 512 threads = 16 warps per block
    dim3 gridDim((N + 63) / 64, (N + 63) / 64);  // Grid based on 64×64 tiles per block

    matmulWMMAKernel<<<gridDim, blockDim>>>(d_A_fp16, d_B_fp16, d_C_fp32, N);
    cudaCheckError(cudaGetLastError());

    // Copy result back (WMMA outputs to FP32)
    cudaCheckError(cudaMemcpy(d_C, d_C_fp32, N * N * sizeof(float), cudaMemcpyDeviceToDevice));
}

// Destructor: Free FP16 buffers
MatmulWMMA::~MatmulWMMA() {
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_fp32);
}
