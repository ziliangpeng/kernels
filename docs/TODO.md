# TODO

## Profiling Tools Setup
- [ ] **Enable Nsight Compute (ncu)** - Get detailed kernel profiling working
  - Currently blocked: `ERR_NVGPUCTRPERM` - no permission to access GPU performance counters
  - Contact system admin to enable profiling:
    ```bash
    sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
    # Or permanently: echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
    ```
  - Once enabled, can use ncu for:
    - SM utilization analysis
    - Memory bandwidth profiling
    - Warp efficiency metrics
    - Instruction throughput analysis
  - Reference: https://developer.nvidia.com/ERR_NVGPUCTRPERM

## CUPTI Integration
- [ ] Try using CUPTI (CUDA Profiling Tools Interface) directly
  - Explore Callback API for hooking into CUDA API calls
  - Test Activity API for collecting GPU events asynchronously
  - Experiment with Metrics API for hardware counters
  - Check CUPTI samples in `$CUDA_HOME/extras/CUPTI/samples/`
  - References:
    - [CUPTI Documentation](https://docs.nvidia.com/cupti/)
    - [CUPTI Tutorial GitHub](https://github.com/eunomia-bpf/cupti-tutorial)
    - [How to build a CUDA profiler](https://conless.dev/blog/2025/cupti-profiler/)

## General CUDA Techniques (applicable to any program)
- [ ] **CUDA Events** - Add accurate GPU timing
  - Replace CPU timing with cudaEvent_t
  - Measure kernel execution time precisely
  - Report GPU time separately from CPU overhead
- [ ] **NVTX Markers** - Add profiling annotations
  - Mark different phases (allocation, transfer, compute)
  - Better visualization in nsys timeline
  - Add range labels for operations
- [ ] **Block Size Tuning** - Experiment with thread counts
  - Add `--threads` flag (128/256/512/1024)
  - Compare performance across different sizes
  - Find optimal configuration for H100
- [ ] **Unified Memory** - Try simpler memory model
  - Replace explicit cudaMalloc/cudaMemcpy with cudaMallocManaged
  - Compare performance vs explicit transfers
  - Understand migration behavior
- [ ] **Streams** - Add concurrent execution
  - Use multiple CUDA streams
  - Overlap computation with memory transfers
  - Measure performance improvement
- [ ] **Warmup Runs** - Skip cold start overhead
  - Add flag for warmup iterations
  - Report statistics excluding warmup
- [ ] **Benchmark Mode** - Multiple iterations with statistics
  - Run N iterations
  - Report min/max/mean/stddev
  - Output CSV format for analysis
- [ ] **CPU vs GPU Comparison** - Baseline performance
  - Implement CPU version
  - Compare execution time
  - Calculate speedup factor

## Algorithm Improvements
- [ ] Add more vector operations (dot product, SAXPY, element-wise multiply)
- [ ] Implement reduction kernels
- [ ] Add matrix multiplication
- [ ] Measure memory bandwidth utilization

## Code Quality
- [ ] Add more command-line flags (iterations, output format)
- [ ] Experiment with different memory patterns (coalesced vs uncoalesced)
- [ ] Better error messages and validation
- [ ] Add unit tests
