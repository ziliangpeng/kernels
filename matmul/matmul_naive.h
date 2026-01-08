#ifndef MATMUL_NAIVE_H
#define MATMUL_NAIVE_H

#include "matmul_kernel.h"

// Naive matrix multiplication - simple triple-nested loop
//
// This implementation uses a straightforward approach where each thread computes
// one output element C[row][col] by computing the dot product of row from A
// and column from B.
//
// MEMORY ACCESS PATTERN:
// - A matrix: Coalesced reads (threads in same warp access consecutive columns)
// - B matrix: Strided reads (threads access elements N apart) - BAD for performance!
// - C matrix: Coalesced writes
//
// PERFORMANCE:
// This achieves only ~1-2% of theoretical peak due to poor B matrix access pattern.
// Use tiled/shared memory version for better performance.

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
