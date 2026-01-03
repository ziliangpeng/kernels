#ifndef SOFTMAX_CUDNN_H
#define SOFTMAX_CUDNN_H

#include "softmax_kernel.h"
#include <cudnn.h>

// cuDNN-based softmax implementation
//
// Uses NVIDIA's cuDNN (CUDA Deep Neural Network) library for production-grade
// softmax computation. cuDNN is the industry standard deep learning primitives
// library used by PyTorch, TensorFlow, and all major frameworks.
//
// Architecture: Single cuDNN API call
// - cudnnSoftmaxForward() handles everything internally
// - Highly optimized for various tensor layouts and batch sizes
// - Automatically selects best algorithm for hardware
//
// Comparison with our implementations:
// - Our CUB/fused: Educational, shows how to build from primitives
// - cuDNN: Production-ready, black-box optimized
//
// Expected Performance:
// - Best or tied for best (NVIDIA's top engineers' work)
// - May use advanced techniques (warp specialization, better occupancy)
// - Optimized for all GPU architectures (Pascal -> Hopper)
//
// cuDNN Softmax Modes:
// - CUDNN_SOFTMAX_MODE_INSTANCE: Softmax across all elements (our use case)
// - CUDNN_SOFTMAX_MODE_CHANNEL: Softmax per channel (for CNNs)
//
// cuDNN Softmax Algorithms:
// - CUDNN_SOFTMAX_FAST: May use approximations, very fast
// - CUDNN_SOFTMAX_ACCURATE: Numerically stable (uses log-sum-exp trick)
// - CUDNN_SOFTMAX_LOG: Computes log(softmax(x)) directly
//
// We use CUDNN_SOFTMAX_ACCURATE + CUDNN_SOFTMAX_MODE_INSTANCE for fair comparison.
//
// Requirements:
// - CUDA 11.0+ with cuDNN 8.0+
// - cuDNN library linked at build time
//
// Class-based interface for accurate profiling
class CudnnSoftmax : public SoftmaxKernel {
private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t tensor_desc;
    int n;

public:
    // Constructor: Allocate cuDNN handle and tensor descriptor
    CudnnSoftmax(int n, int threadsPerBlock);

    // Execute: Pure kernel execution (no setup/teardown overhead)
    void execute(const float *d_input, float *d_output) override;

    // Destructor: Free cuDNN resources
    ~CudnnSoftmax() override;
};

// Legacy C-style API (for backwards compatibility)
float softmax_Cudnn(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
