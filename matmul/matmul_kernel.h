#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

// Base interface for all matrix multiplication kernel implementations
//
// PURPOSE:
// This interface separates workspace allocation (constructor/destructor) from
// kernel execution (execute method), enabling accurate profiling of pure
// kernel performance without setup/teardown overhead.
//
// USAGE PATTERN:
// 1. Constructor: Allocate workspace (device buffers, library handles, etc.)
// 2. execute(): Pure kernel execution - ONLY THIS IS TIMED in benchmarks
// 3. Destructor: Free workspace (automatic via RAII)
//
// MEMORY ASSUMPTIONS:
// - d_A, d_B, and d_C are already allocated on GPU by caller
// - No host-device transfers in execute()
// - Workspace is managed by implementation (allocated in constructor, freed in destructor)
//
// EXAMPLE:
// ```cpp
// // Outside timing loop:
// MatmulNaive kernel(N, blockDim);  // Allocates workspace if needed
//
// // Inside timing loop (ONLY this is timed):
// for (int i = 0; i < 100; i++) {
//     cudaEventRecord(start);
//     kernel.execute(d_A, d_B, d_C);  // Pure kernel execution!
//     cudaEventRecord(stop);
// }
//
// // After timing loop:
// // Destructor automatically frees workspace
// ```
class MatmulKernel {
public:
    virtual ~MatmulKernel() {}

    // Execute matrix multiplication kernel: C = A × B (pure computation, no setup/teardown)
    //
    // Parameters:
    //   d_A: Device pointer to matrix A (N×N, row-major layout, already on GPU)
    //   d_B: Device pointer to matrix B (N×N, row-major layout, already on GPU)
    //   d_C: Device pointer to output matrix C (N×N, row-major layout, already on GPU)
    //
    // This method should ONLY launch kernels - no allocation, no transfers.
    virtual void execute(const float *d_A, const float *d_B, float *d_C) = 0;
};

#endif
