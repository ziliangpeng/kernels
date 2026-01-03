#include "softmax_cudnn.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// CUDNN SOFTMAX: Industry-Standard Deep Learning Primitive
// ============================================================================
//
// WHAT IS CUDNN?
// --------------
// cuDNN (CUDA Deep Neural Network library) is NVIDIA's production-grade
// library for deep learning primitives. It's the backend for:
//   - PyTorch (torch.nn.functional.softmax)
//   - TensorFlow (tf.nn.softmax)
//   - JAX, MXNet, Caffe, and virtually all frameworks
//
// cuDNN softmax is a SINGLE FUNCTION CALL that handles everything:
//   - Optimal algorithm selection based on hardware
//   - Numerically stable computation (log-sum-exp trick)
//   - Support for various tensor layouts (NCHW, NHWC, etc.)
//   - Batched operations
//   - Tuned for all GPU architectures (Pascal → Hopper)
//
// CUDNN API PATTERN
// -----------------
// cuDNN uses "descriptors" to describe tensor shapes and operations:
//
// 1. Create handle: cudnnCreate(&handle)
// 2. Create tensor descriptor: cudnnCreateTensorDescriptor(&desc)
// 3. Set tensor descriptor: cudnnSetTensor4dDescriptor(desc, format, datatype, n, c, h, w)
// 4. Execute operation: cudnnSoftmaxForward(handle, algo, mode, alpha, desc, input, beta, desc, output)
// 5. Cleanup: cudnnDestroyTensorDescriptor(desc), cudnnDestroy(handle)
//
// TENSOR DESCRIPTOR FOR 1D SOFTMAX
// ---------------------------------
// Our input is 1D array of n elements, but cuDNN expects 4D tensors (N, C, H, W).
// We map our 1D array to 4D as: (1, 1, 1, n)
//   - N (batch size) = 1
//   - C (channels) = 1
//   - H (height) = 1
//   - W (width) = n (our array length)
//
// SOFTMAX MODES
// -------------
// CUDNN_SOFTMAX_MODE_INSTANCE: Softmax across all elements (our use case)
//   softmax(x[i]) = exp(x[i]) / sum(exp(x[j])) for all j
//
// CUDNN_SOFTMAX_MODE_CHANNEL: Softmax per channel (for CNNs)
//   softmax(x[n,c,h,w]) = exp(x[n,c,h,w]) / sum_c(exp(x[n,c,h,w]))
//
// SOFTMAX ALGORITHMS
// ------------------
// CUDNN_SOFTMAX_FAST: May sacrifice some accuracy for speed
// CUDNN_SOFTMAX_ACCURATE: Numerically stable (log-sum-exp), our choice
// CUDNN_SOFTMAX_LOG: Computes log(softmax(x)) directly
//
// ALPHA/BETA SCALING
// ------------------
// output = alpha * softmax(input) + beta * output
// For simple softmax: alpha=1.0, beta=0.0
//
// EXPECTED PERFORMANCE
// --------------------
// cuDNN should be fastest or tied for fastest because:
// 1. Written by NVIDIA's top performance engineers
// 2. Uses advanced techniques (warp specialization, better occupancy)
// 3. Optimized and tuned for every GPU architecture
// 4. Production-tested across millions of training runs
//
// Comparison with our implementations:
//   - Our implementations: Educational, transparent, ~0.2-0.3ms for 1M elements
//   - cuDNN: Black-box, production-grade, expected similar or better
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
// CLASS-BASED IMPLEMENTATION: Separates setup from execution
// ============================================================================

// Constructor: Allocate cuDNN handle and configure tensor descriptor
CudnnSoftmax::CudnnSoftmax(int n, int threadsPerBlock) : n(n) {
    // Note: threadsPerBlock is unused for cuDNN (it decides internally)
    (void)threadsPerBlock;

    // Create cuDNN handle
    cudnnCheckError(cudnnCreate(&cudnn));

    // Create and configure tensor descriptor
    // Map 1D array [n] to 4D tensor (1, 1, 1, n)
    cudnnCheckError(cudnnCreateTensorDescriptor(&tensor_desc));
    cudnnCheckError(cudnnSetTensor4dDescriptor(
        tensor_desc,
        CUDNN_TENSOR_NCHW,      // Format: N, C, H, W
        CUDNN_DATA_FLOAT,        // Data type: float32
        1,                       // N (batch size) = 1
        1,                       // C (channels) = 1
        1,                       // H (height) = 1
        n                        // W (width) = n elements
    ));
}

// Execute: Pure kernel execution (ONLY this is timed in benchmarks)
void CudnnSoftmax::execute(const float *d_input, float *d_output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnCheckError(cudnnSoftmaxForward(
        cudnn,
        CUDNN_SOFTMAX_ACCURATE,        // Algorithm: numerically stable
        CUDNN_SOFTMAX_MODE_INSTANCE,   // Mode: softmax across all elements
        &alpha,
        tensor_desc,
        d_input,
        &beta,
        tensor_desc,
        d_output
    ));
}

// Destructor: Free cuDNN resources
CudnnSoftmax::~CudnnSoftmax() {
    cudnnDestroyTensorDescriptor(tensor_desc);
    cudnnDestroy(cudnn);
}

// ============================================================================
// LEGACY C-STYLE API: Wrapper for backwards compatibility
// ============================================================================

float softmax_Cudnn(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    CudnnSoftmax kernel(n, threadsPerBlock);
    kernel.execute(d_input, d_output);
    return 0.0f;  // Timing handled by caller
}
