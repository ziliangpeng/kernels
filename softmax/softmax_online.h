#ifndef SOFTMAX_ONLINE_H
#define SOFTMAX_ONLINE_H

// Online softmax - single-pass streaming computation (SKELETON - not implemented)
//
// Single-pass algorithm that updates running statistics as data is read.
// Most memory-efficient approach.
//
// Algorithm:
//   Initialize: running_max = -infinity, running_sum = 0
//   For each element x[i]:
//     old_max = running_max
//     running_max = max(running_max, x[i])
//     running_sum = running_sum * exp(old_max - running_max) + exp(x[i] - running_max)
//
// Returns execution time in milliseconds
float softmax_Online(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
