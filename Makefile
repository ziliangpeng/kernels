# Makefile for CUDA programs

# CUDA compiler
NVCC = nvcc

# Compiler flags (can be overridden: make NVCC_FLAGS="-O2 -arch=sm_86")
NVCC_FLAGS ?= -O2 -arch=sm_90

# Targets
TARGETS = vector mem_benchmark

# Default target - build all
all: $(TARGETS)

# Build vector from multiple source files
vector: vector.cu vector_kernels.cu vector_init.cu cuda_utils.cu cuda_utils.h vector_kernels.h vector_init.h
	$(NVCC) $(NVCC_FLAGS) vector.cu vector_kernels.cu vector_init.cu cuda_utils.cu -o vector

# Build mem_benchmark
mem_benchmark: mem_benchmark.cu cuda_utils.h
	$(NVCC) $(NVCC_FLAGS) mem_benchmark.cu -o mem_benchmark

# Run vector
run: vector
	./vector

# Test all binaries
test: all
	@echo "=========================================="
	@echo "Running verification tests..."
	@echo "=========================================="
	@echo "Testing vector - ADD mode..."
	@./vector -n 1000 --mode add -v > /dev/null && echo "  ✓ ADD: 1K elements"
	@./vector -n 1000000 --mode add -v > /dev/null && echo "  ✓ ADD: 1M elements"
	@./vector -n 10000000 --mode add -v > /dev/null && echo "  ✓ ADD: 10M elements"
	@echo ""
	@echo "Testing vector - VMA mode..."
	@./vector -n 10000 --mode vma -v > /dev/null && echo "  ✓ VMA separate: 10K elements"
	@./vector -n 10000 --mode vma --fused -v > /dev/null && echo "  ✓ VMA fused: 10K elements"
	@./vector -n 10000 --mode vma --fused --vectorized -v > /dev/null && echo "  ✓ VMA fused+vectorized: 10K elements (n%4==0)"
	@./vector -n 1000000 --mode vma --fused -v > /dev/null && echo "  ✓ VMA fused: 1M elements"
	@echo ""
	@echo "Testing vector - Different block sizes..."
	@./vector -n 100000 -b 128 -v > /dev/null && echo "  ✓ Block size 128"
	@./vector -n 100000 -b 512 -v > /dev/null && echo "  ✓ Block size 512"
	@echo ""
	@echo "Testing mem_benchmark..."
	@./mem_benchmark -n 10 -i 1 > /dev/null && echo "  ✓ Pageable (10MB)"
	@./mem_benchmark -n 10 -p -i 1 > /dev/null && echo "  ✓ Pinned (10MB)"
	@echo ""
	@echo "=========================================="
	@echo "All tests passed!"
	@echo "=========================================="

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o

# Check if CUDA is available
check:
	@echo "Checking CUDA installation..."
	@which nvcc >/dev/null || (echo "ERROR: nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvcc --version

.PHONY: all run test clean check
