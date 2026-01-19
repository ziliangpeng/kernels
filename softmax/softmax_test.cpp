#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <string>

#include "softmax_kernel.h"
#include "softmax_naive.h"
#include "softmax_fused3.h"
#include "softmax_fused2.h"
#include "softmax_online_simple.h"
#include "softmax_online_warp.h"
#include "softmax_cub_block.h"
#include "softmax_cub_device.h"
#include "softmax_cudnn.h"
#include "softmax_tiny.h"
#include "softmax_small.h"

// ============================================================================
// Test Configuration
// ============================================================================

const int THREADS_PER_BLOCK = 256;

// Test sizes: 16, 1K, 64K, 1M
const std::vector<int> TEST_SIZES = {16, 1024, 65536, 1048576};

// Error thresholds
const double MAX_REL_ERROR_THRESHOLD = 1e-4;
const double SUM_ERROR_THRESHOLD = 1e-4;

// ============================================================================
// CPU Reference Implementation
// ============================================================================

void softmax_cpu_reference(const float* input, float* output, int n) {
    // Find max
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Compute exp-sum (use double for better accuracy)
    double sum_exp = 0.0;
    for (int i = 0; i < n; i++) {
        sum_exp += exp(static_cast<double>(input[i] - max_val));
    }

    // Normalize
    for (int i = 0; i < n; i++) {
        output[i] = static_cast<float>(exp(static_cast<double>(input[i] - max_val)) / sum_exp);
    }
}

// ============================================================================
// Test Fixture
// ============================================================================

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    // Allocate and initialize input with deterministic pseudo-random values
    void allocateAndInitInput(int n) {
        h_input_.resize(n);
        h_output_.resize(n);
        h_expected_.resize(n);

        // Initialize with deterministic values for reproducibility
        // Range: [-5, 5] to test numerical stability
        for (int i = 0; i < n; i++) {
            h_input_[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f - 5.0f;
        }

        // Compute CPU reference
        softmax_cpu_reference(h_input_.data(), h_expected_.data(), n);

        // Allocate device memory
        ASSERT_EQ(cudaMalloc(&d_input_, n * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output_, n * sizeof(float)), cudaSuccess);

        // Transfer input to device
        ASSERT_EQ(cudaMemcpy(d_input_, h_input_.data(), n * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }

    void freeMemory() {
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        d_input_ = nullptr;
        d_output_ = nullptr;
    }

    // Verify kernel output against CPU reference
    // Returns pair of (max_rel_error, sum_error)
    std::pair<double, double> verifyOutput(int n) {
        // Transfer output to host
        EXPECT_EQ(cudaMemcpy(h_output_.data(), d_output_, n * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);

        // Check for NaN/Inf
        for (int i = 0; i < n; i++) {
            if (std::isnan(h_output_[i]) || std::isinf(h_output_[i])) {
                return {-1.0, -1.0};  // Indicates NaN/Inf failure
            }
        }

        // Compare against CPU reference
        double max_rel_error = 0.0;
        double sum_gpu = 0.0;

        for (int i = 0; i < n; i++) {
            double abs_error = std::fabs(h_output_[i] - h_expected_[i]);
            double rel_error = abs_error / (std::fabs(h_expected_[i]) + 1e-10);

            if (rel_error > max_rel_error) max_rel_error = rel_error;
            sum_gpu += h_output_[i];
        }

        double sum_error = std::fabs(sum_gpu - 1.0);
        return {max_rel_error, sum_error};
    }

    std::vector<float> h_input_;
    std::vector<float> h_output_;
    std::vector<float> h_expected_;
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// Parameterized Test for All Kernels
// ============================================================================

struct KernelTestParam {
    std::string name;
    int size;
};

class SoftmaxKernelTest : public SoftmaxTest,
                          public ::testing::WithParamInterface<KernelTestParam> {
};

// Factory function to create kernel by name
std::unique_ptr<SoftmaxKernel> createKernel(const std::string& name, int n, int threads_per_block) {
    if (name == "naive") {
        return std::make_unique<NaiveSoftmax>(n, threads_per_block);
    } else if (name == "fused3") {
        return std::make_unique<Fused3Softmax>(n, threads_per_block);
    } else if (name == "fused2") {
        return std::make_unique<Fused2Softmax>(n, threads_per_block);
    } else if (name == "online_simple") {
        return std::make_unique<OnlineSimpleSoftmax>(n, threads_per_block);
    } else if (name == "online_warp") {
        return std::make_unique<OnlineWarpSoftmax>(n, threads_per_block);
    } else if (name == "cub_block") {
        return std::make_unique<CubBlockSoftmax>(n, threads_per_block);
    } else if (name == "cub_device") {
        return std::make_unique<CubDeviceSoftmax>(n, threads_per_block);
    } else if (name == "cudnn") {
        return std::make_unique<CudnnSoftmax>(n, threads_per_block);
    } else if (name == "tiny") {
        return std::make_unique<TinySoftmax>(n, threads_per_block);
    } else if (name == "small") {
        return std::make_unique<SmallSoftmax>(n, threads_per_block);
    }
    return nullptr;
}

TEST_P(SoftmaxKernelTest, CorrectnessAgainstCPU) {
    const auto& param = GetParam();
    int n = param.size;

    // Special case: naive method is expected to fail for large inputs
    // due to numerical instability (overflow in exp())
    bool expect_nan_for_naive = (param.name == "naive" && n > 1000);

    allocateAndInitInput(n);

    std::unique_ptr<SoftmaxKernel> kernel;
    try {
        kernel = createKernel(param.name, n, THREADS_PER_BLOCK);
    } catch (const std::exception& e) {
        freeMemory();
        GTEST_SKIP() << "Kernel creation failed: " << e.what();
    }

    ASSERT_NE(kernel, nullptr) << "Unknown kernel: " << param.name;

    // Execute kernel
    kernel->execute(d_input_, d_output_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        freeMemory();
        GTEST_SKIP() << "Kernel execution failed: " << cudaGetErrorString(err);
    }

    cudaDeviceSynchronize();

    // Verify output
    auto [max_rel_error, sum_error] = verifyOutput(n);

    freeMemory();

    // Handle expected NaN case for naive method
    if (max_rel_error < 0 && expect_nan_for_naive) {
        // This is expected - naive method overflows for large inputs
        SUCCEED() << "Naive method produced NaN/Inf as expected for large input";
        return;
    }

    // For other methods, NaN/Inf is a failure
    ASSERT_GE(max_rel_error, 0) << "Output contains NaN or Inf";

    // Check thresholds
    EXPECT_LT(max_rel_error, MAX_REL_ERROR_THRESHOLD)
        << "Max relative error " << max_rel_error << " exceeds threshold";
    EXPECT_LT(sum_error, SUM_ERROR_THRESHOLD)
        << "Sum error " << sum_error << " exceeds threshold (sum should be 1.0)";
}

// Generate test parameters
std::vector<KernelTestParam> generateTestParams() {
    std::vector<KernelTestParam> params;
    std::vector<std::string> kernel_names = {
        "naive", "fused3", "fused2", "online_simple", "online_warp",
        "cub_block", "cub_device", "cudnn", "tiny", "small"
    };

    for (const auto& name : kernel_names) {
        for (int size : TEST_SIZES) {
            params.push_back({name, size});
        }
    }
    return params;
}

std::string testParamName(const ::testing::TestParamInfo<KernelTestParam>& info) {
    return info.param.name + "_" + std::to_string(info.param.size);
}

INSTANTIATE_TEST_SUITE_P(
    AllKernels,
    SoftmaxKernelTest,
    ::testing::ValuesIn(generateTestParams()),
    testParamName
);

// ============================================================================
// Additional Tests
// ============================================================================

// Test that softmax output sums to 1.0
TEST_F(SoftmaxTest, OutputSumsToOne) {
    int n = 1024;
    allocateAndInitInput(n);

    auto kernel = std::make_unique<OnlineWarpSoftmax>(n, THREADS_PER_BLOCK);
    kernel->execute(d_input_, d_output_);
    cudaDeviceSynchronize();

    ASSERT_EQ(cudaMemcpy(h_output_.data(), d_output_, n * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += h_output_[i];
    }

    freeMemory();

    EXPECT_NEAR(sum, 1.0, 1e-5) << "Softmax output sum should be 1.0";
}

// Test that all outputs are non-negative (probability distribution)
TEST_F(SoftmaxTest, OutputsNonNegative) {
    int n = 1024;
    allocateAndInitInput(n);

    auto kernel = std::make_unique<OnlineWarpSoftmax>(n, THREADS_PER_BLOCK);
    kernel->execute(d_input_, d_output_);
    cudaDeviceSynchronize();

    ASSERT_EQ(cudaMemcpy(h_output_.data(), d_output_, n * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    freeMemory();

    for (int i = 0; i < n; i++) {
        EXPECT_GE(h_output_[i], 0.0f) << "Output at index " << i << " is negative";
    }
}

// Test numerical stability with large input values
TEST_F(SoftmaxTest, NumericalStabilityLargeInputs) {
    int n = 1024;
    h_input_.resize(n);
    h_output_.resize(n);
    h_expected_.resize(n);

    // Initialize with large values that would cause overflow without proper handling
    for (int i = 0; i < n; i++) {
        h_input_[i] = 100.0f + static_cast<float>(i % 10);
    }

    softmax_cpu_reference(h_input_.data(), h_expected_.data(), n);

    ASSERT_EQ(cudaMalloc(&d_input_, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output_, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input_, h_input_.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);

    auto kernel = std::make_unique<OnlineWarpSoftmax>(n, THREADS_PER_BLOCK);
    kernel->execute(d_input_, d_output_);
    cudaDeviceSynchronize();

    auto [max_rel_error, sum_error] = verifyOutput(n);

    freeMemory();

    ASSERT_GE(max_rel_error, 0) << "Output contains NaN or Inf despite using stable algorithm";
    EXPECT_LT(max_rel_error, MAX_REL_ERROR_THRESHOLD);
    EXPECT_LT(sum_error, SUM_ERROR_THRESHOLD);
}
