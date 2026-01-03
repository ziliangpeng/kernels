#ifndef SOFTMAX_NAIVE_H
#define SOFTMAX_NAIVE_H

#include "softmax_kernel.h"

// Naive (unstable) softmax - for demonstration of overflow issues
//
// This implementation computes exp(x) directly without max subtraction,
// which causes numerical overflow for large input values.

// Class-based interface for accurate profiling
class NaiveSoftmax : public SoftmaxKernel {
private:
    float *d_workspace;  // Workspace for recursive reduction
    int n, threadsPerBlock;
    int max_workspace_size;

public:
    // Constructor: Allocate workspace for reduction stages
    NaiveSoftmax(int n, int threadsPerBlock);

    // Execute: Pure kernel execution (no setup/teardown overhead)
    void execute(const float *d_input, float *d_output) override;

    // Destructor: Free workspace
    ~NaiveSoftmax() override;
};

// Legacy C-style API (for backwards compatibility)
float softmax_Naive(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
