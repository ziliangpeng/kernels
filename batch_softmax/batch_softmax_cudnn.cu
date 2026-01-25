#include "batch_softmax_cudnn.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// CUDNN BATCH SOFTMAX: Industry-Standard Deep Learning Primitive
// ============================================================================
//
// This implementation uses cuDNN for batched softmax computation.
// Each row of the (batch_size, dim) matrix gets independent softmax.
//
// TENSOR DESCRIPTOR FOR BATCHED SOFTMAX
// -------------------------------------
// We need softmax applied independently to each row (batch element).
// cuDNN tensor layout: (N, C, H, W) with NCHW format
//
// Mapping: (batch_size, dim) -> (batch_size, dim, 1, 1)
//   - N = batch_size (number of rows)
//   - C = dim (softmax dimension)
//   - H = 1
//   - W = 1
//
// With CUDNN_SOFTMAX_MODE_CHANNEL, softmax is applied across C (dim) for each N.
// This gives us independent softmax for each row, exactly what we need.
//
// ============================================================================

// Helper macro for cuDNN error checking
#define cudnnCheckError(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN Error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

CudnnBatchSoftmax::CudnnBatchSoftmax(int batch_size, int dim, int algorithm)
    : batch_size(batch_size), dim(dim) {
    // Set algorithm: 0 = FAST, 1 = ACCURATE (default)
    algo = (algorithm == 0) ? CUDNN_SOFTMAX_FAST : CUDNN_SOFTMAX_ACCURATE;

    // Create cuDNN handle
    cudnnCheckError(cudnnCreate(&cudnn));

    // Create and configure tensor descriptor
    // Map (batch_size, dim) to 4D tensor (batch_size, dim, 1, 1)
    // Softmax over C (channel) dimension = dim
    cudnnCheckError(cudnnCreateTensorDescriptor(&tensor_desc));
    cudnnCheckError(cudnnSetTensor4dDescriptor(
        tensor_desc,
        CUDNN_TENSOR_NCHW,       // Format: N, C, H, W
        CUDNN_DATA_FLOAT,        // Data type: float32
        batch_size,              // N (batch size)
        dim,                     // C (channels = softmax dimension)
        1,                       // H (height) = 1
        1                        // W (width) = 1
    ));
}

void CudnnBatchSoftmax::execute(const float *d_input, float *d_output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnCheckError(cudnnSoftmaxForward(
        cudnn,
        algo,                        // Algorithm: FAST or ACCURATE
        CUDNN_SOFTMAX_MODE_CHANNEL,  // Mode: softmax across C (dim) for each N
        &alpha,
        tensor_desc,
        d_input,
        &beta,
        tensor_desc,
        d_output
    ));
}

CudnnBatchSoftmax::~CudnnBatchSoftmax() {
    cudnnDestroyTensorDescriptor(tensor_desc);
    cudnnDestroy(cudnn);
}
