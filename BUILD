load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")
load("@rules_python//python:defs.bzl", "py_binary")

# ============================================================================
# Shared CUDA Libraries (used across multiple projects)
# ============================================================================

# Utility functions for CUDA (error checking, statistics)
cuda_library(
    name = "cuda_utils",
    srcs = ["cuda_utils.cu"],
    hdrs = ["cuda_utils.h"],
    target_compatible_with = ["//constraints:cuda"],
    visibility = ["//cuda:__subpackages__"],  # Visible to cuda/* subdirs
)

# Vector initialization utilities
cuda_library(
    name = "vector_init",
    srcs = ["vector_init.cu"],
    hdrs = ["vector_init.h"],
    deps = [":cuda_utils"],
    target_compatible_with = ["//constraints:cuda"],
    visibility = ["//cuda:__subpackages__"],
)

# Reduction kernels (shared by reduce.cu and softmax)
cuda_library(
    name = "reduce_kernels",
    srcs = ["reduce_kernels.cu"],
    hdrs = ["reduce_kernels.h"],
    deps = [":cuda_utils"],
    target_compatible_with = ["//constraints:cuda"],
    visibility = ["//cuda:__subpackages__"],
)

# ============================================================================
# Root-level CUDA Projects (not yet restructured)
# ============================================================================

# Element-wise kernels (shared across multiple modules)
cuda_library(
    name = "elementwise_kernels",
    srcs = ["elementwise_kernels.cu"],
    hdrs = ["elementwise_kernels.h"],
    target_compatible_with = ["//constraints:cuda"],
    visibility = ["//visibility:public"],
)

# Vector addition binary
cuda_binary(
    name = "vector",
    srcs = ["vector.cu"],
    deps = [
        ":cuda_utils",
        ":vector_init",
        ":elementwise_kernels",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

# Reduction binary
cuda_binary(
    name = "reduce",
    srcs = ["reduce.cu"],
    deps = [
        ":cuda_utils",
        ":vector_init",
        ":reduce_kernels",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

# Memory benchmark binary
cuda_binary(
    name = "mem_benchmark",
    srcs = ["mem_benchmark.cu"],
    deps = [":cuda_utils"],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

# ============================================================================
# Convenience Targets
# ============================================================================

# Build all root-level binaries (for backwards compatibility with Makefile)
filegroup(
    name = "all_root_binaries",
    srcs = [
        ":vector",
        ":reduce",
        ":mem_benchmark",
    ],
)

# ============================================================================
# Python Benchmarks (Bazel-managed deps)
# ============================================================================

py_binary(
    name = "tinygrad_comparison",
    srcs = ["tinygrad_comparison.py"],
    deps = [
        "@pip//numpy",
        "@pip//tinygrad",
    ],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)

py_binary(
    name = "run_comparison",
    srcs = ["run_comparison.py"],
    data = [
        ":vector",
        ":reduce",
        ":tinygrad_comparison",
        "//cuda/softmax:softmax",
    ],
    deps = ["@bazel_tools//tools/python/runfiles"],
    target_compatible_with = ["//constraints:cuda"],
    tags = ["cuda", "gpu"],
)
