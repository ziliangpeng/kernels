#include "matmul_vectorized.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Vectorized Memory Access kernel (Kernel 6)
// Uses float4 (128-bit) loads from global memory
// Transposes A tile in shared memory for coalesced SMEM loads

#define BM_VEC 128
#define BN_VEC 128
#define BK_VEC 8
#define TM_VEC 8
#define TN_VEC 8

// Threads per block: (BM/TM) * (BN/TN) = 16 * 16 = 256
#define NUM_THREADS_VEC ((BM_VEC / TM_VEC) * (BN_VEC / TN_VEC))

__global__ void matmulVectorizedKernel(const float *A, const float *B, float *C, int N) {
    // Shared memory: As is transposed for coalesced access
    // As[BK][BM] instead of As[BM][BK]
    __shared__ float As[BK_VEC][BM_VEC];
    __shared__ float Bs[BK_VEC][BN_VEC];

    // Thread position in the output tile grid
    const int threadCol = threadIdx.x % (BN_VEC / TN_VEC);  // 0-15
    const int threadRow = threadIdx.x / (BN_VEC / TN_VEC);  // 0-15

    // Block position in output
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Move pointers to the block's starting position
    A += blockRow * BM_VEC * N;
    B += blockCol * BN_VEC;
    C += blockRow * BM_VEC * N + blockCol * BN_VEC;

    // Thread results: each thread computes TM x TN elements
    float threadResults[TM_VEC][TN_VEC] = {{0.0f}};

    // Registers for A and B values
    float regA[TM_VEC];
    float regB[TN_VEC];

    // For loading A: 256 threads, BM*BK = 128*8 = 1024 elements
    // Each thread loads 4 elements using float4
    // We need 1024/4 = 256 float4 loads total, so 1 per thread
    const int innerRowA = threadIdx.x / (BK_VEC / 4);  // 256 / 2 = 128 (row in A)
    const int innerColA = threadIdx.x % (BK_VEC / 4);  // 0-1 (which float4 in the row)

    // For loading B: BK*BN = 8*128 = 1024 elements
    // Each thread loads 4 elements using float4
    const int innerRowB = threadIdx.x / (BN_VEC / 4);  // 256 / 32 = 8 (row in B)
    const int innerColB = threadIdx.x % (BN_VEC / 4);  // 0-31 (which float4 in the row)

    // Loop over K dimension in chunks of BK
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_VEC) {
        // Load tile of A using float4, storing transposed into As[BK][BM]
        // A is BM x BK, we want As[k][m] = A[m][k]
        // Thread loads A[innerRowA][tileIdx + innerColA*4 : innerColA*4+4]
        if (innerRowA < BM_VEC && blockRow * BM_VEC + innerRowA < N) {
            float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
            if (tileIdx + innerColA * 4 + 3 < N) {
                tmp = *reinterpret_cast<const float4*>(&A[innerRowA * N + innerColA * 4]);
            } else {
                // Handle boundary
                for (int i = 0; i < 4; i++) {
                    if (tileIdx + innerColA * 4 + i < N) {
                        reinterpret_cast<float*>(&tmp)[i] = A[innerRowA * N + innerColA * 4 + i];
                    }
                }
            }
            // Store transposed: As[k][m] where k = innerColA*4+i, m = innerRowA
            As[innerColA * 4 + 0][innerRowA] = tmp.x;
            As[innerColA * 4 + 1][innerRowA] = tmp.y;
            As[innerColA * 4 + 2][innerRowA] = tmp.z;
            As[innerColA * 4 + 3][innerRowA] = tmp.w;
        }

        // Load tile of B using float4
        if (innerRowB < BK_VEC && tileIdx + innerRowB < N) {
            float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
            if (blockCol * BN_VEC + innerColB * 4 + 3 < N) {
                tmp = *reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4]);
            } else {
                // Handle boundary
                for (int i = 0; i < 4; i++) {
                    if (blockCol * BN_VEC + innerColB * 4 + i < N) {
                        reinterpret_cast<float*>(&tmp)[i] = B[innerRowB * N + innerColB * 4 + i];
                    }
                }
            }
            Bs[innerRowB][innerColB * 4 + 0] = tmp.x;
            Bs[innerRowB][innerColB * 4 + 1] = tmp.y;
            Bs[innerRowB][innerColB * 4 + 2] = tmp.z;
            Bs[innerRowB][innerColB * 4 + 3] = tmp.w;
        }

        __syncthreads();

        // Move pointers for next iteration
        A += BK_VEC;
        B += BK_VEC * N;

        // Compute using outer product
        // As is now transposed, so As[k][m] gives us what we need
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_VEC; dotIdx++) {
            // Load A values from transposed shared memory (coalesced!)
            #pragma unroll
            for (int i = 0; i < TM_VEC; i++) {
                regA[i] = As[dotIdx][threadRow * TM_VEC + i];
            }

            // Load B row into registers
            #pragma unroll
            for (int j = 0; j < TN_VEC; j++) {
                regB[j] = Bs[dotIdx][threadCol * TN_VEC + j];
            }

            // Outer product
            #pragma unroll
            for (int i = 0; i < TM_VEC; i++) {
                #pragma unroll
                for (int j = 0; j < TN_VEC; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory using float4
    #pragma unroll
    for (int i = 0; i < TM_VEC; i++) {
        int globalRow = blockRow * BM_VEC + threadRow * TM_VEC + i;
        if (globalRow < N) {
            #pragma unroll
            for (int j = 0; j < TN_VEC; j += 4) {
                int globalCol = blockCol * BN_VEC + threadCol * TN_VEC + j;
                if (globalCol + 3 < N) {
                    float4 tmp;
                    tmp.x = threadResults[i][j + 0];
                    tmp.y = threadResults[i][j + 1];
                    tmp.z = threadResults[i][j + 2];
                    tmp.w = threadResults[i][j + 3];
                    *reinterpret_cast<float4*>(&C[(threadRow * TM_VEC + i) * N + threadCol * TN_VEC + j]) = tmp;
                } else {
                    // Handle boundary
                    for (int k = 0; k < 4 && globalCol + k < N; k++) {
                        C[(threadRow * TM_VEC + i) * N + threadCol * TN_VEC + j + k] = threadResults[i][j + k];
                    }
                }
            }
        }
    }
}

MatmulVectorized::MatmulVectorized(int N, int blockDim) : N(N), blockDim(blockDim) {
    // No workspace needed
}

void MatmulVectorized::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_VEC);
    dim3 blocks((N + BN_VEC - 1) / BN_VEC,
                (N + BM_VEC - 1) / BM_VEC);

    matmulVectorizedKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaCheckError(cudaGetLastError());
}

MatmulVectorized::~MatmulVectorized() {
    // No workspace to free
}
