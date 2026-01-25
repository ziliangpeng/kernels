// C API for softmax_online_warp CUDA kernel
// This provides a simple C interface that can be wrapped by pybind11

#ifndef SOFTMAX_C_API_H
#define SOFTMAX_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the softmax kernel instance
typedef struct SoftmaxKernelHandle* SoftmaxHandle;

// Create a new softmax kernel instance
// Returns NULL on failure
SoftmaxHandle softmax_create(int n, int threads_per_block);

// Execute softmax on device memory
// d_input and d_output must be device pointers
void softmax_execute(SoftmaxHandle handle, const float* d_input, float* d_output);

// Destroy the kernel instance and free resources
void softmax_destroy(SoftmaxHandle handle);

// Allocate device memory
float* softmax_alloc_device(int n);

// Free device memory
void softmax_free_device(float* d_ptr);

// Copy host to device
void softmax_copy_to_device(float* d_dst, const float* h_src, int n);

// Copy device to host
void softmax_copy_to_host(float* h_dst, const float* d_src, int n);

// Synchronize device
void softmax_sync();

// CUDA event timing
typedef struct CudaEventHandle* CudaEvent;
CudaEvent softmax_event_create();
void softmax_event_destroy(CudaEvent event);
void softmax_event_record(CudaEvent event);
void softmax_event_sync(CudaEvent event);
float softmax_event_elapsed(CudaEvent start, CudaEvent end);

#ifdef __cplusplus
}
#endif

#endif // SOFTMAX_C_API_H
