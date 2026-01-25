"""CUDA-related Bazel macros.

This module provides macros for building Python targets that depend on
pip-installed NVIDIA CUDA libraries (like nvidia-cudnn-cu12, etc.).
"""

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

def _nvidia_lib_dirs():
    """Returns a list of (package, path) tuples for NVIDIA library directories.

    The paths are relative to the runfiles root and point to the lib/
    directories inside each NVIDIA pip package.
    """
    return [
        ("nvidia_cublas_cu12", "nvidia/cublas/lib"),
        ("nvidia_cuda_cupti_cu12", "nvidia/cuda_cupti/lib"),
        ("nvidia_cuda_nvrtc_cu12", "nvidia/cuda_nvrtc/lib"),
        ("nvidia_cuda_runtime_cu12", "nvidia/cuda_runtime/lib"),
        ("nvidia_cudnn_cu12", "nvidia/cudnn/lib"),
        ("nvidia_cufft_cu12", "nvidia/cufft/lib"),
        ("nvidia_curand_cu12", "nvidia/curand/lib"),
        ("nvidia_cusolver_cu12", "nvidia/cusolver/lib"),
        ("nvidia_cusparse_cu12", "nvidia/cusparse/lib"),
        ("nvidia_cusparselt_cu12", "cusparselt/lib"),
        ("nvidia_nccl_cu12", "nvidia/nccl/lib"),
        ("nvidia_nvjitlink_cu12", "nvidia/nvjitlink/lib"),
        ("nvidia_nvtx_cu12", "nvidia/nvtx/lib"),
    ]

def _generate_cuda_wrapper(name, package_path, nvidia_packages):
    """Generates a shell wrapper script that sets LD_LIBRARY_PATH.

    The wrapper finds NVIDIA lib directories in runfiles and sets
    LD_LIBRARY_PATH before executing the actual Python binary.

    Args:
        name: The name of the binary target
        package_path: The Bazel package path (e.g., "cuda/softmax")
        nvidia_packages: List of NVIDIA package names that need library paths
    """
    # Generate the library path construction code
    lib_paths = []
    for pkg, subpath in _nvidia_lib_dirs():
        if pkg in nvidia_packages:
            lib_paths.append('    "$$RUNFILES_DIR/_main/../rules_python++pip+pip_cuda_313_{pkg}/site-packages/{subpath}"'.format(
                pkg = pkg,
                subpath = subpath,
            ))

    paths_code = " \\\n".join(lib_paths) if lib_paths else '    ""'

    # Build the full path to the impl binary in runfiles
    if package_path:
        impl_path = package_path + "/" + name + "_impl"
    else:
        impl_path = name + "_impl"

    # Note: All $ must be escaped as $$ for Bazel genrule cmd attribute
    # All { and } must be escaped as {{ and }} for .format()
    return '''#!/bin/bash
# Auto-generated wrapper script for CUDA Python binary
# Sets LD_LIBRARY_PATH to include NVIDIA pip package libraries

set -e

# Find runfiles directory
if [[ -n "$${{RUNFILES_DIR:-}}" ]]; then
    : # already set
elif [[ -d "$${{0}}.runfiles" ]]; then
    RUNFILES_DIR="$${{0}}.runfiles"
elif [[ -d "$${{BASH_SOURCE[0]}}.runfiles" ]]; then
    RUNFILES_DIR="$${{BASH_SOURCE[0]}}.runfiles"
else
    echo "Error: Could not find runfiles directory" >&2
    exit 1
fi

# Build LD_LIBRARY_PATH from NVIDIA pip packages
NVIDIA_LIB_PATHS=(
{paths_code}
)

# Prepend our library paths (not append) to ensure they take precedence
NEW_LD_PATH=""
for lib_dir in "$${{NVIDIA_LIB_PATHS[@]}}"; do
    if [[ -d "$$lib_dir" ]]; then
        NEW_LD_PATH="$${{NEW_LD_PATH}}:$$lib_dir"
    fi
done
# Remove leading colon and prepend to existing LD_LIBRARY_PATH
NEW_LD_PATH="$${{NEW_LD_PATH#:}}"
if [[ -n "$${{LD_LIBRARY_PATH:-}}" ]]; then
    export LD_LIBRARY_PATH="$$NEW_LD_PATH:$$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$$NEW_LD_PATH"
fi

# Execute the actual Python binary
exec "$$RUNFILES_DIR/_main/{impl_path}" "$$@"
'''.format(impl_path = impl_path, paths_code = paths_code)

def cuda_py_binary(
        name,
        srcs,
        deps = [],
        data = [],
        main = None,
        nvidia_packages = None,
        **kwargs):
    """A py_binary wrapper that sets up LD_LIBRARY_PATH for NVIDIA CUDA libraries.

    This macro creates two targets:
    1. {name}_impl - the actual Python binary
    2. {name} - a shell wrapper that sets LD_LIBRARY_PATH and runs the impl

    Args:
        name: The name of the binary target
        srcs: Python source files
        deps: Python dependencies
        data: Data dependencies
        main: Main Python file (defaults to first src)
        nvidia_packages: List of NVIDIA package names that need library paths.
            If None, auto-detects from deps.
        **kwargs: Additional arguments passed to py_binary
    """

    # Auto-detect NVIDIA packages from deps if not specified
    if nvidia_packages == None:
        nvidia_packages = []
        known_nvidia_pkgs = [pkg for pkg, _ in _nvidia_lib_dirs()]
        for dep in deps:
            for pkg in known_nvidia_pkgs:
                if pkg in dep:
                    nvidia_packages.append(pkg)

    # Determine the main file for the impl binary
    impl_main = main
    if impl_main == None and len(srcs) == 1:
        impl_main = srcs[0]

    # Create the actual Python binary with _impl suffix
    py_binary(
        name = name + "_impl",
        srcs = srcs,
        deps = deps,
        data = data,
        main = impl_main,
        **kwargs
    )

    # Generate wrapper script content
    wrapper_content = _generate_cuda_wrapper(name, native.package_name(), nvidia_packages)

    # Create the wrapper script using a genrule
    native.genrule(
        name = name + "_wrapper_gen",
        outs = [name + "_wrapper.sh"],
        cmd = "cat > $@ << 'WRAPPER_EOF'\n" + wrapper_content + "WRAPPER_EOF",
        visibility = ["//visibility:private"],
    )

    # Create the wrapper binary
    native.sh_binary(
        name = name,
        srcs = [name + "_wrapper.sh"],
        data = [name + "_impl"],
        target_compatible_with = kwargs.get("target_compatible_with", []),
        visibility = kwargs.get("visibility", None),
    )
