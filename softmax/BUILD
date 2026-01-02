load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary", "cuda_test")

# ============================================================================
# Softmax Kernels Library
# ============================================================================

cuda_library(
    name = "softmax_kernels",
    srcs = ["softmax_kernels.cu"],
    hdrs = ["softmax_kernels.h"],
    deps = [
        "//cuda:cuda_utils",      # Reference parent shared lib
        "//cuda:reduce_kernels",  # Reference parent shared lib
    ],
    includes = [".."],  # Add parent directory to include path
    visibility = ["//cuda/softmax:__subpackages__"],
)

# ============================================================================
# Softmax Main Binary
# ============================================================================

cuda_binary(
    name = "softmax",
    srcs = ["softmax.cu"],
    deps = [
        ":softmax_kernels",
        "//cuda:cuda_utils",
        "//cuda:vector_init",
    ],
    tags = ["cuda", "gpu"],
)

# ============================================================================
# Test Targets (mirrors Makefile test cases)
# ============================================================================
# IMPORTANT: tags = ["cuda", "gpu"] allows CI to skip these (no GPU in GitHub Actions)

# Naive method tests
cuda_test(
    name = "softmax_naive_1k_test",
    srcs = ["softmax.cu"],
    args = ["--size", "1000", "--method", "naive", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "softmax_naive_10k_test",
    srcs = ["softmax.cu"],
    args = ["--size", "10000", "--method", "naive", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

# Multi-pass method tests
cuda_test(
    name = "softmax_multi_1k_test",
    srcs = ["softmax.cu"],
    args = ["--size", "1000", "--method", "multi", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "softmax_multi_100k_test",
    srcs = ["softmax.cu"],
    args = ["--size", "100000", "--method", "multi", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "softmax_multi_1m_test",
    srcs = ["softmax.cu"],
    args = ["--size", "1000000", "--method", "multi", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

# Fused method tests
cuda_test(
    name = "softmax_fused_1k_test",
    srcs = ["softmax.cu"],
    args = ["--size", "1000", "--method", "fused", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "softmax_fused_100k_test",
    srcs = ["softmax.cu"],
    args = ["--size", "100000", "--method", "fused", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "softmax_fused_1m_test",
    srcs = ["softmax.cu"],
    args = ["--size", "1000000", "--method", "fused", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "softmax_fused_10m_test",
    srcs = ["softmax.cu"],
    args = ["--size", "10000000", "--method", "fused", "--verify"],
    deps = [":softmax_kernels", "//cuda:cuda_utils", "//cuda:vector_init"],
    tags = ["cuda", "gpu"],
    timeout = "moderate",  # Larger test needs more time
)

# ============================================================================
# Test Suite (run all softmax tests)
# ============================================================================

test_suite(
    name = "all_tests",
    tests = [
        ":softmax_naive_1k_test",
        ":softmax_naive_10k_test",
        ":softmax_multi_1k_test",
        ":softmax_multi_100k_test",
        ":softmax_multi_1m_test",
        ":softmax_fused_1k_test",
        ":softmax_fused_100k_test",
        ":softmax_fused_1m_test",
        ":softmax_fused_10m_test",
    ],
)
