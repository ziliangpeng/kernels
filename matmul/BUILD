load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary", "cuda_test")

# ============================================================================
# Matrix Multiplication Implementation Libraries
# ============================================================================

# Matrix initialization utilities
cuda_library(
    name = "matrix_init",
    srcs = ["matrix_init.cu"],
    hdrs = ["matrix_init.h"],
    deps = [
        "//cuda:cuda_utils",
    ],
    target_compatible_with = ["//constraints:cuda"],
    includes = [".."],
)

# Base interface
cc_library(
    name = "matmul_kernel_h",
    hdrs = ["matmul_kernel.h"],
)

# Naive matrix multiplication implementation
cuda_library(
    name = "matmul_naive",
    srcs = ["matmul_naive.cu"],
    hdrs = [
        "matmul_naive.h",
        "matmul_kernel.h",  # Base interface
    ],
    deps = [
        "//cuda:cuda_utils",
    ],
    target_compatible_with = ["//constraints:cuda"],
    includes = [".."],
)

# ============================================================================
# Matrix Multiplication Main Binary
# ============================================================================

cuda_binary(
    name = "matmul",
    srcs = ["matmul.cpp"],
    deps = [
        ":matmul_naive",
        ":matmul_kernel_h",
        ":matrix_init",
        "//cuda:cuda_utils",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

# ============================================================================
# Test Targets
# ============================================================================
# IMPORTANT: tags = ["cuda", "gpu"] allows CI to skip these (no GPU in GitHub Actions)

# Test naive implementation on small matrices
cuda_test(
    name = "matmul_naive_64_test",
    srcs = ["matmul.cpp"],
    args = ["--size", "64", "--method", "naive", "--verify"],
    deps = [
        ":matmul_naive",
        ":matmul_kernel_h",
        ":matrix_init",
        "//cuda:cuda_utils",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "matmul_naive_128_test",
    srcs = ["matmul.cpp"],
    args = ["--size", "128", "--method", "naive", "--verify"],
    deps = [
        ":matmul_naive",
        ":matmul_kernel_h",
        ":matrix_init",
        "//cuda:cuda_utils",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "matmul_naive_256_test",
    srcs = ["matmul.cpp"],
    args = ["--size", "256", "--method", "naive", "--verify"],
    deps = [
        ":matmul_naive",
        ":matmul_kernel_h",
        ":matrix_init",
        "//cuda:cuda_utils",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

cuda_test(
    name = "matmul_naive_512_test",
    srcs = ["matmul.cpp"],
    args = ["--size", "512", "--method", "naive", "--verify"],
    deps = [
        ":matmul_naive",
        ":matmul_kernel_h",
        ":matrix_init",
        "//cuda:cuda_utils",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

# ============================================================================
# Test Suite (run all matmul tests)
# ============================================================================

test_suite(
    name = "all_tests",
    tests = [
        ":matmul_naive_64_test",
        ":matmul_naive_128_test",
        ":matmul_naive_256_test",
        ":matmul_naive_512_test",
    ],
)
