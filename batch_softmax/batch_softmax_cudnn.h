#ifndef BATCH_SOFTMAX_CUDNN_H
#define BATCH_SOFTMAX_CUDNN_H

#include "batch_softmax_kernel.h"
#include <cudnn.h>

// cuDNN-based batch softmax implementation
//
// Uses NVIDIA's cuDNN library for production-grade batched softmax computation.
// cuDNN is the industry standard deep learning primitives library used by
// PyTorch, TensorFlow, and all major frameworks.
//
// Architecture: Single cuDNN API call
// - cudnnSoftmaxForward() handles everything internally
// - Highly optimized for various tensor layouts and batch sizes
// - Automatically selects best algorithm for hardware
//
// Tensor descriptor for batched softmax:
// - Shape: (batch_size, dim, 1, 1) with NCHW format
// - CUDNN_SOFTMAX_MODE_CHANNEL computes softmax across C dimension (dim)
// - Each row of batch_size rows gets independent softmax
//
// cuDNN Softmax Algorithms:
// - CUDNN_SOFTMAX_FAST: May use approximations, very fast
// - CUDNN_SOFTMAX_ACCURATE: Numerically stable (uses log-sum-exp trick)
// - CUDNN_SOFTMAX_LOG: Computes log(softmax(x)) directly
//
// We use CUDNN_SOFTMAX_ACCURATE + CUDNN_SOFTMAX_MODE_CHANNEL for batched softmax.
class CudnnBatchSoftmax : public BatchSoftmaxKernel {
public:
    CudnnBatchSoftmax(int batch_size, int dim, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~CudnnBatchSoftmax() override;

private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t tensor_desc;
    int batch_size;
    int dim;
};

#endif  // BATCH_SOFTMAX_CUDNN_H
