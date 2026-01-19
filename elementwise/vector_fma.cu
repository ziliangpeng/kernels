#include "vector_fma.h"

// CUDA kernel for fused multiply-add (FMA)
// Computes d[i] = a[i] * b[i] + c[i] in a single kernel
__global__ void vectorFMA(const float *a, const float *b, const float *c, float *d, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        d[idx] = a[idx] * b[idx] + c[idx];
    }
}

// Vectorized FMA kernel using float4
// Each thread processes 4 elements at once using vectorized loads/stores
__global__ void vectorFMA_float4(const float *a, const float *b, const float *c, float *d, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread processes 4 elements
    // Note: n must be multiple of 4 (validated in host code)
    if (idx < n/4) {
        // Load 4 floats at once (single memory transaction)
        float4 va = ((float4*)a)[idx];
        float4 vb = ((float4*)b)[idx];
        float4 vc = ((float4*)c)[idx];

        // Compute 4 results
        float4 vd;
        vd.x = va.x * vb.x + vc.x;
        vd.y = va.y * vb.y + vc.y;
        vd.z = va.z * vb.z + vc.z;
        vd.w = va.w * vb.w + vc.w;

        // Store 4 floats at once (single memory transaction)
        ((float4*)d)[idx] = vd;
    }
}
