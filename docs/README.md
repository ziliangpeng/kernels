# My First CUDA Program

This directory contains a simple CUDA vector addition program - the classic "Hello World" of GPU programming!

## What it does

The program:
1. Creates two arrays of 1 million floating-point numbers
2. Copies them to GPU memory
3. Adds them element-by-element in parallel using GPU threads
4. Copies the result back to CPU
5. Verifies the result is correct

## Prerequisites

You need the NVIDIA CUDA Toolkit installed. Check if you have it:

```bash
nvcc --version
```

If not installed, you may need to load it as a module (on HPC systems):

```bash
module load cuda
```

Or install it from: https://developer.nvidia.com/cuda-downloads

## How to compile and run

### Using Make (recommended)

```bash
# Check CUDA installation
make check

# Compile the program
make

# Run it
./vector

# Or compile and run in one command
make run

# Clean up
make clean
```

### Manual compilation

```bash
nvcc -O2 -arch=sm_90 vector.cu vector_kernels.cu vector_init.cu -o vector
./vector
```

Note: You may need to adjust `-arch=sm_90` based on your GPU's compute capability.

## Expected output

```
Vector addition of 1048576 elements
Launching kernel with 4096 blocks and 256 threads per block
Verifying result...
Test PASSED! All 1048576 elements correctly computed.

Congratulations! Your first CUDA program ran successfully!
```

## Understanding the code

### Key CUDA concepts demonstrated:

1. **Kernel function** (`__global__`): Runs on the GPU
   ```cuda
   __global__ void vectorAdd(...)
   ```

2. **Thread indexing**: Each thread calculates which element to process
   ```cuda
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   ```

3. **Memory management**:
   - `cudaMalloc()` - Allocate GPU memory
   - `cudaMemcpy()` - Copy data between CPU and GPU
   - `cudaFree()` - Free GPU memory

4. **Kernel launch**: Execute the kernel with specified grid/block dimensions
   ```cuda
   vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(...)
   ```

5. **Error checking**: Always check for CUDA errors!

## Next steps

Try modifying the program to:
- Change the array size
- Implement other operations (subtraction, multiplication, dot product)
- Measure execution time with CUDA events
- Compare GPU vs CPU performance

## Troubleshooting

- **No CUDA device found**: Make sure you're on a machine with an NVIDIA GPU
- **Compilation errors**: Check your CUDA toolkit version and GPU architecture
- **Wrong results**: Ensure your GPU compute capability matches the `-arch` flag
