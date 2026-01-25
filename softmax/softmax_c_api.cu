// C API implementation for softmax_online_warp CUDA kernel

#include "softmax_c_api.h"
#include "softmax_online_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Opaque handle wrapping the C++ class
struct SoftmaxKernelHandle {
    OnlineWarpSoftmax* kernel;
};

struct CudaEventHandle {
    cudaEvent_t event;
};

extern "C" {

SoftmaxHandle softmax_create(int n, int threads_per_block) {
    SoftmaxHandle handle = new SoftmaxKernelHandle;
    handle->kernel = new OnlineWarpSoftmax(n, threads_per_block);
    return handle;
}

void softmax_execute(SoftmaxHandle handle, const float* d_input, float* d_output) {
    handle->kernel->execute(d_input, d_output);
}

void softmax_destroy(SoftmaxHandle handle) {
    delete handle->kernel;
    delete handle;
}

float* softmax_alloc_device(int n) {
    float* d_ptr;
    cudaCheckError(cudaMalloc(&d_ptr, n * sizeof(float)));
    return d_ptr;
}

void softmax_free_device(float* d_ptr) {
    cudaFree(d_ptr);
}

void softmax_copy_to_device(float* d_dst, const float* h_src, int n) {
    cudaCheckError(cudaMemcpy(d_dst, h_src, n * sizeof(float), cudaMemcpyHostToDevice));
}

void softmax_copy_to_host(float* h_dst, const float* d_src, int n) {
    cudaCheckError(cudaMemcpy(h_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

void softmax_sync() {
    cudaCheckError(cudaDeviceSynchronize());
}

CudaEvent softmax_event_create() {
    CudaEvent handle = new CudaEventHandle;
    cudaCheckError(cudaEventCreate(&handle->event));
    return handle;
}

void softmax_event_destroy(CudaEvent event) {
    cudaEventDestroy(event->event);
    delete event;
}

void softmax_event_record(CudaEvent event) {
    cudaCheckError(cudaEventRecord(event->event));
}

void softmax_event_sync(CudaEvent event) {
    cudaCheckError(cudaEventSynchronize(event->event));
}

float softmax_event_elapsed(CudaEvent start, CudaEvent end) {
    float ms;
    cudaCheckError(cudaEventElapsedTime(&ms, start->event, end->event));
    return ms;
}

}  // extern "C"
