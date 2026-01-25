// pybind11 wrapper for softmax CUDA kernel using C API
//
// This module exposes the OnlineWarpSoftmax CUDA kernel to Python.
// The interface uses NumPy arrays for input/output.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "softmax_c_api.h"

namespace py = pybind11;

// RAII wrappers for automatic resource cleanup
struct DeviceBuffer {
    float* ptr;
    explicit DeviceBuffer(int n) : ptr(softmax_alloc_device(n)) {}
    ~DeviceBuffer() { if (ptr) softmax_free_device(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct KernelHandle {
    SoftmaxHandle handle;
    KernelHandle(int n, int block_size) : handle(softmax_create(n, block_size)) {}
    ~KernelHandle() { if (handle) softmax_destroy(handle); }
    KernelHandle(const KernelHandle&) = delete;
    KernelHandle& operator=(const KernelHandle&) = delete;
};

struct EventHandle {
    CudaEvent event;
    EventHandle() : event(softmax_event_create()) {}
    ~EventHandle() { if (event) softmax_event_destroy(event); }
    EventHandle(const EventHandle&) = delete;
    EventHandle& operator=(const EventHandle&) = delete;
};

// Helper to validate input and get buffer info
std::tuple<int, float*> validate_input(py::array_t<float>& input) {
    py::buffer_info buf_in = input.request();
    if (buf_in.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    return {static_cast<int>(buf_in.shape[0]), static_cast<float*>(buf_in.ptr)};
}

// Helper to copy result back to NumPy array
py::array_t<float> copy_result_to_numpy(float* d_output, int n) {
    auto result = py::array_t<float>(n);
    py::buffer_info buf_out = result.request();
    softmax_copy_to_host(static_cast<float*>(buf_out.ptr), d_output, n);
    return result;
}

// Softmax using OnlineWarpSoftmax kernel
py::array_t<float> softmax_cuda(py::array_t<float> input, int block_size = 256) {
    auto [n, host_input] = validate_input(input);

    DeviceBuffer d_input(n);
    DeviceBuffer d_output(n);
    softmax_copy_to_device(d_input.ptr, host_input, n);

    KernelHandle kernel(n, block_size);
    softmax_execute(kernel.handle, d_input.ptr, d_output.ptr);
    softmax_sync();

    return copy_result_to_numpy(d_output.ptr, n);
}

// Timed softmax for benchmarking (returns output and time in ms)
std::tuple<py::array_t<float>, float> softmax_cuda_timed(py::array_t<float> input, int block_size = 256) {
    auto [n, host_input] = validate_input(input);

    DeviceBuffer d_input(n);
    DeviceBuffer d_output(n);
    softmax_copy_to_device(d_input.ptr, host_input, n);

    KernelHandle kernel(n, block_size);
    EventHandle start, stop;

    softmax_event_record(start.event);
    softmax_execute(kernel.handle, d_input.ptr, d_output.ptr);
    softmax_event_record(stop.event);
    softmax_event_sync(stop.event);

    float time_ms = softmax_event_elapsed(start.event, stop.event);
    return {copy_result_to_numpy(d_output.ptr, n), time_ms};
}

// Benchmark function: run kernel multiple times and return list of times
std::vector<float> softmax_cuda_benchmark(py::array_t<float> input, int iterations = 100, int warmup = 10, int block_size = 256) {
    auto [n, host_input] = validate_input(input);

    DeviceBuffer d_input(n);
    DeviceBuffer d_output(n);
    softmax_copy_to_device(d_input.ptr, host_input, n);

    KernelHandle kernel(n, block_size);
    EventHandle start, stop;

    // Warmup iterations
    for (int i = 0; i < warmup; i++) {
        softmax_execute(kernel.handle, d_input.ptr, d_output.ptr);
    }
    softmax_sync();

    // Benchmark iterations with per-iteration timing
    std::vector<float> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        softmax_event_record(start.event);
        softmax_execute(kernel.handle, d_input.ptr, d_output.ptr);
        softmax_event_record(stop.event);
        softmax_event_sync(stop.event);
        times.push_back(softmax_event_elapsed(start.event, stop.event));
    }

    return times;
}

PYBIND11_MODULE(softmax_cuda, m) {
    m.doc() = "CUDA Softmax (OnlineWarp) - pybind11 bindings";

    m.def("softmax", &softmax_cuda,
          "Compute softmax using CUDA OnlineWarp kernel",
          py::arg("input"),
          py::arg("block_size") = 256);

    m.def("softmax_timed", &softmax_cuda_timed,
          "Compute softmax and return (output, time_ms) tuple",
          py::arg("input"),
          py::arg("block_size") = 256);

    m.def("benchmark", &softmax_cuda_benchmark,
          "Run softmax kernel multiple times and return list of timing results (ms)",
          py::arg("input"),
          py::arg("iterations") = 100,
          py::arg("warmup") = 10,
          py::arg("block_size") = 256);
}
