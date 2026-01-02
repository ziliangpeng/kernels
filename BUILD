load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")

# ============================================================================
# Shared CUDA Libraries (used across multiple projects)
# ============================================================================

# Utility functions for CUDA (error checking, statistics)
cuda_library(
    name = "cuda_utils",
    srcs = ["cuda_utils.cu"],
    hdrs = ["cuda_utils.h"],
    visibility = ["//cuda:__subpackages__"],  # Visible to cuda/* subdirs
)

# Vector initialization utilities
cuda_library(
    name = "vector_init",
    srcs = ["vector_init.cu"],
    hdrs = ["vector_init.h"],
    deps = [":cuda_utils"],
    visibility = ["//cuda:__subpackages__"],
)

# Reduction kernels (shared by reduce.cu and softmax)
cuda_library(
    name = "reduce_kernels",
    srcs = ["reduce_kernels.cu"],
    hdrs = ["reduce_kernels.h"],
    deps = [":cuda_utils"],
    visibility = ["//cuda:__subpackages__"],
)

# ============================================================================
# Root-level CUDA Projects (not yet restructured)
# ============================================================================

# Vector kernels (only used by vector.cu)
cuda_library(
    name = "vector_kernels",
    srcs = ["vector_kernels.cu"],
    hdrs = ["vector_kernels.h"],
)

# Vector addition binary
cuda_binary(
    name = "vector",
    srcs = ["vector.cu"],
    deps = [
        ":cuda_utils",
        ":vector_init",
        ":vector_kernels",
    ],
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
    tags = ["cuda", "gpu"],
)

# Memory benchmark binary
cuda_binary(
    name = "mem_benchmark",
    srcs = ["mem_benchmark.cu"],
    deps = [":cuda_utils"],
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
