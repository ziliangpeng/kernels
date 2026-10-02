#ifndef MATMUL_NAIVE_H
#define MATMUL_NAIVE_H

#include "matmul_kernel.h"

// Naive matrix multiplication - simple triple-nested loop
//
// This implementation uses a straightforward approach where each thread computes
// one output element C[row][col] by computing the dot product of row from A
// and column from B.
//
// MEMORY ACCESS PATTERN (deliberately uncoalesced, siboehm kernel 1):
// threadIdx.x -> row, threadIdx.y -> col. Lanes of a warp share col (when
// blockDim.x >= 32) and differ in row.
// - A matrix: A[row * N + k] — lanes N*4 bytes apart -> one 32B sector per lane (strided, BAD)
// - B matrix: B[k * N + col] — same address across lanes sharing col (broadcast)
// - C matrix: C[row * N + col] — strided writes
//
// matmul_coalesced swaps the mapping (threadIdx.x -> col) to make B/C
// accesses contiguous and A a broadcast.

// Class-based interface for accurate profiling
class MatmulNaive : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension (e.g., 16 for 16×16 blocks)

public:
    // Constructor: Store configuration (no workspace needed for naive kernel)
    MatmulNaive(int N, int blockDim);

    // Execute: Pure kernel execution (no setup/teardown overhead)
    void execute(const float *d_A, const float *d_B, float *d_C) override;

    // Destructor: Nothing to free
    ~MatmulNaive() override;
};

#endif
