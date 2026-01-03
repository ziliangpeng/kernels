#ifndef SOFTMAX_ONLINE_SIMPLE_H
#define SOFTMAX_ONLINE_SIMPLE_H

#include "softmax_kernel.h"

// Online Softmax - Simple Thread-Level Implementation (Educational)
//
// This implementation demonstrates the elegant online softmax algorithm using
// a straightforward 2-kernel approach with thread-level state management.
//
// Algorithm: Streaming computation of max and sum
// ------------------------------------------------
// Instead of multiple passes over data (find max, then compute exp-sum, then normalize),
// we maintain running statistics that update as we process each element:
//
// Initialize: running_max = -∞, running_sum = 0
//
// For each element x[i]:
//   old_max = running_max
//   running_max = max(running_max, x[i])
//   running_sum = running_sum * exp(old_max - running_max) + exp(x[i] - running_max)
//
// Final: output[i] = exp(x[i] - running_max) / running_sum
//
// Key Property: The adjustment factor exp(old_max - new_max) keeps the sum
// numerically stable when the max changes.
//
// Architecture: 2-Kernel Approach
// --------------------------------
// Kernel 1: Block-level online statistics
//   - Each thread maintains local (max, sum) state
//   - Processes multiple elements via grid-stride loop
//   - Reduces threads to single (block_max, block_sum) using shared memory
//   - Critical: Must correctly merge (max, sum) pairs during reduction
//
// Kernel 2: Global reduce + normalize
//   - Reduces all block stats to global (max, sum)
//   - Normalizes output using global scalars
//
// Educational Focus:
// ------------------
// - Clear demonstration of online update formulas
// - Shows how to merge (max, sum) pairs during reduction
// - Simple enough to understand and debug
// - Good introduction to streaming algorithms
//
// Performance Characteristics:
// ----------------------------
// - Expected: ~0.08-0.12ms for 100K elements
// - Slower than Fused3 (~0.038ms) due to more exp() calls in merge operations
// - But demonstrates elegant single-pass concept
//
// Requirements:
// - CUDA 11.0+
// - C++11 or later

// Class-based interface for accurate profiling
class OnlineSimpleSoftmax : public SoftmaxKernel {
private:
    float *d_block_maxes, *d_block_sums;
    float *d_global_max, *d_global_sum;
    int n, threadsPerBlock, numBlocks;

public:
    // Constructor: Allocate intermediate buffers
    OnlineSimpleSoftmax(int n, int threadsPerBlock);

    // Execute: Pure kernel execution (no setup/teardown overhead)
    void execute(const float *d_input, float *d_output) override;

    // Destructor: Free intermediate buffers
    ~OnlineSimpleSoftmax() override;
};

// Legacy C-style API (for backwards compatibility)
float softmax_OnlineSimple(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
