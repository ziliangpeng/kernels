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
#include <cstring>
#include <cuda_bf16.h>

// FP32 traits: lets the SAME templated kernel serve as its own FP32 control
// (same code path, zero conversion) — clean same-binary A/B for autotune.
struct DTypeTraitsFloat {
    using T = float;
    static constexpr const char *name = "fp32";
    static constexpr const char *suffix = "fp32";
    using VecT = float4;  // 16-byte vector: 4 floats
    __device__ static float to_float(T x) { return x; }
    __host__ __device__ static T from_float(float x) { return x; }
    __host__ __device__ static T zero() { return 0.0f; }
    __host__ __device__ static void zero_vec(VecT *v) { *v = make_float4(0, 0, 0, 0); }
    __host__ __device__ static T get_vec_elem(const VecT &v, int i) {
        return (i == 0) ? v.x : (i == 1) ? v.y : (i == 2) ? v.z : v.w;
    }
    __host__ __device__ static void set_vec_elem(VecT *v, int i, T x) {
        if (i == 0) v->x = x; else if (i == 1) v->y = x;
        else if (i == 2) v->z = x; else v->w = x;
    }
};

struct DTypeTraitsHalf {
    using T = __half;
    static constexpr const char *name = "fp16";
    static constexpr const char *suffix = "f16";
    using VecT = uint4;   // 16-byte vector: 8 halves
    __device__ static float to_float(T x) { return __half2float(x); }
    __host__ __device__ static T from_float(float x) { return __float2half(x); }
    __host__ __device__ static T zero() { return from_float(0.0f); }
    __host__ __device__ static void zero_vec(VecT *v) { *v = make_uint4(0, 0, 0, 0); }
    __host__ __device__ static T get_vec_elem(const VecT &v, int i) {
        // 8 halves packed in 4 uints (2 per uint)
        uint u = (i & 1) ? (v.x >> 16) : (v.x & 0xFFFFu);
        if (i >= 2) u = (i & 1) ? (v.y >> 16) : (v.y & 0xFFFFu);
        if (i >= 4) u = (i & 1) ? (v.z >> 16) : (v.z & 0xFFFFu);
        if (i >= 6) u = (i & 1) ? (v.w >> 16) : (v.w & 0xFFFFu);
        __half h;
        memcpy(&h, &u, 2);
        return h;
    }
    __host__ __device__ static void set_vec_elem(VecT *v, int i, T x) {
        uint u; memcpy(&u, &x, 2);
        uint *slot = (i < 2) ? &v->x : (i < 4) ? &v->y : (i < 6) ? &v->z : &v->w;
        if (i & 1) *slot = (*slot & 0xFFFFu) | (u << 16);
        else       *slot = (*slot & 0xFFFF0000u) | u;
    }
};

struct DTypeTraitsBf16 {
    using T = __nv_bfloat16;
    static constexpr const char *name = "bf16";
    static constexpr const char *suffix = "bf16";
    __device__ static float to_float(T x) { return __bfloat162float(x); }
    __host__ __device__ static T from_float(float x) { return __float2bfloat16(x); }

    __host__ __device__ static T zero() { return from_float(0.0f); }
};

// Convert a float device buffer to a T device buffer (device-side kernel).
// Defined in matmul_naive_typed.cu (must be compiled by nvcc; host compilers
// have no CUDA built-ins).
template <typename Traits>
__global__ void convertFp32ToKernel(const float *in, typename Traits::T *out, int n);

#endif // DTYPE_TRAITS_H
