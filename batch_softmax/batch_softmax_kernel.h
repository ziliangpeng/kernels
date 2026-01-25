#ifndef BATCH_SOFTMAX_KERNEL_H
#define BATCH_SOFTMAX_KERNEL_H

// Base interface for all batch softmax kernel implementations
//
// PURPOSE:
// This interface separates workspace allocation (constructor/destructor) from
// kernel execution (execute method), enabling accurate profiling of pure
// kernel performance without setup/teardown overhead.
//
// BATCH SOFTMAX:
// Computes softmax independently over each row of a 2D matrix.
// Input: (batch_size, dim) matrix where each row is a separate softmax instance
// Output: (batch_size, dim) matrix where each row sums to 1.0
//
// USAGE PATTERN:
// 1. Constructor: Allocate workspace (device buffers, etc.)
// 2. execute(): Pure kernel execution - ONLY THIS IS TIMED in benchmarks
// 3. Destructor: Free workspace (automatic via RAII)
//
// MEMORY ASSUMPTIONS:
// - d_input and d_output are already allocated on GPU by caller
// - Both are contiguous row-major arrays of size (batch_size * dim)
// - No host-device transfers in execute()
// - Workspace is managed by implementation (allocated in constructor, freed in destructor)
//
// EXAMPLE:
// ```cpp
// // Outside timing loop:
// NaiveBatchSoftmax kernel(batch_size, dim, threadsPerBlock);
//
// // Inside timing loop (ONLY this is timed):
// for (int i = 0; i < 100; i++) {
//     cudaEventRecord(start);
//     kernel.execute(d_input, d_output);  // Pure kernel execution!
//     cudaEventRecord(stop);
// }
// ```
class BatchSoftmaxKernel {
public:
    virtual ~BatchSoftmaxKernel() {}

    // Execute batch softmax kernel (pure computation, no setup/teardown)
    //
    // Parameters:
    //   d_input: Device pointer to input array (batch_size * dim floats, row-major)
    //   d_output: Device pointer to output array (batch_size * dim floats, row-major)
    //
    // This method should ONLY launch kernels - no allocation, no transfers.
    virtual void execute(const float *d_input, float *d_output) = 0;
};

#endif  // BATCH_SOFTMAX_KERNEL_H
