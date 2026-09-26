#ifndef DTYPE_TRAITS_H
#define DTYPE_TRAITS_H

// Data-type traits for the multi-precision GEMM ladder.
//
// Semantics: 16-bit STORAGE (A and B in half / __nv_bfloat16), FP32 ACCUMULATION.
// Loads convert to float inside the kernel; the multiply-accumulate itself runs
// in FP32 (scalar FMA units). This is the standard no-Tensor-Core 16-bit
// pattern: the win comes from halved GMEM traffic + halved SMEM footprint,
// not from different math units.
//
// The benchmark harness (matrix_init, MatmulKernel interface) is FP32-native:
// host data is float, kernels receive float* buffers. The 16-bit variants
// therefore CONVERT in the constructor (not inside the timed execute()),
// mirroring MatmulWMMA/MatmulCublasBf16's device-side-conversion pattern but
// WITHOUT their flaw of converting inside the timed region.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct DTypeTraitsHalf {
    using T = __half;
    static constexpr const char *name = "fp16";
    static constexpr const char *suffix = "f16";
    __device__ static float to_float(T x) { return __half2float(x); }
    __host__ static T from_float(float x) { return __float2half(x); }
};

struct DTypeTraitsBf16 {
    using T = __nv_bfloat16;
    static constexpr const char *name = "bf16";
    static constexpr const char *suffix = "bf16";
    __device__ static float to_float(T x) { return __bfloat162float(x); }
    __host__ static T from_float(float x) { return __float2bfloat16(x); }
};

// Convert a float device buffer to a T device buffer (device-side kernel).
template <typename Traits>
__global__ void convertFp32ToKernel(const float *in, typename Traits::T *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Traits::from_float(in[i]);
}

#endif // DTYPE_TRAITS_H
