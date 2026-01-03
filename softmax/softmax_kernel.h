#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

// Base interface for all softmax kernel implementations
//
// PURPOSE:
// This interface separates workspace allocation (constructor/destructor) from
// kernel execution (execute method), enabling accurate profiling of pure
// kernel performance without setup/teardown overhead.
//
// USAGE PATTERN:
// 1. Constructor: Allocate workspace (device buffers, cuDNN handles, etc.)
// 2. execute(): Pure kernel execution - ONLY THIS IS TIMED in benchmarks
// 3. Destructor: Free workspace (automatic via RAII)
//
// MEMORY ASSUMPTIONS:
// - d_input and d_output are already allocated on GPU by caller
// - No host-device transfers in execute()
// - Workspace is managed by implementation (allocated in constructor, freed in destructor)
//
// EXAMPLE:
// ```cpp
// // Outside timing loop:
// CudnnSoftmax kernel(n, threadsPerBlock);  // Allocates cuDNN handle, descriptors
//
// // Inside timing loop (ONLY this is timed):
// for (int i = 0; i < 1000; i++) {
//     cudaEventRecord(start);
//     kernel.execute(d_input, d_output);  // Pure kernel execution!
//     cudaEventRecord(stop);
// }
//
// // After timing loop:
// // Destructor automatically frees workspace
// ```
class SoftmaxKernel {
public:
    virtual ~SoftmaxKernel() {}

    // Execute softmax kernel (pure computation, no setup/teardown)
    //
    // Parameters:
    //   d_input: Device pointer to input array (already on GPU)
    //   d_output: Device pointer to output array (already on GPU)
    //
    // This method should ONLY launch kernels - no allocation, no transfers.
    virtual void execute(const float *d_input, float *d_output) = 0;
};

#endif
