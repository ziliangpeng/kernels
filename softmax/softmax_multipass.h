#ifndef SOFTMAX_MULTIPASS_H
#define SOFTMAX_MULTIPASS_H

// Multi-pass (7+ kernels) softmax - numerically stable baseline approach
//
// Uses recursive reduction with multiple kernel launches:
// Stage 1: Find max(x) via recursive reduction
// Stage 2: Compute sum(exp(x - max)) via recursive reduction
// Stage 3: Normalize output
//
// This is the stable baseline but slower due to multiple kernel launches.
//
// Returns execution time in milliseconds
float softmax_MultiPass(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
