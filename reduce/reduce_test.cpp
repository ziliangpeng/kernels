#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "sum_reduce.h"
#include "sum_reduce_atomic.h"
#include "max_reduce.h"

// ============================================================================
// Test Configuration
// ============================================================================

const int THREADS_PER_BLOCK = 256;
const std::vector<int> TEST_SIZES = {16, 1024, 65536, 1048576};
const double SUM_ERROR_THRESHOLD = 1e-4;  // Relaxed for accumulation order differences

// ============================================================================
// Test Fixture
// ============================================================================

class ReduceTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

// ============================================================================
// Sum Reduction Tests
// ============================================================================

class SumReduceTest : public ReduceTest,
                      public ::testing::WithParamInterface<int> {
};

TEST_P(SumReduceTest, GPU_CorrectnessAgainstCPU) {
    int n = GetParam();

    // Allocate and initialize
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f - 5.0f;
    }

    // CPU reference (use double for accuracy)
    double expected = 0.0;
    for (int i = 0; i < n; i++) {
        expected += h_input[i];
    }

    // Allocate device memory
    float *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    // Run GPU reduction
    float result = vectorSum_GPU(d_input, n, THREADS_PER_BLOCK);

    // Verify
    double rel_error = std::fabs(result - expected) / std::fabs(expected);
    printf("[vectorSum_GPU @ %d]: result = %.6f, expected = %.6f, rel_error = %.2e\n",
           n, result, (float)expected, rel_error);
    EXPECT_LT(rel_error, SUM_ERROR_THRESHOLD);

    cudaFree(d_input);
}

TEST_P(SumReduceTest, GPU_Warp_CorrectnessAgainstCPU) {
    int n = GetParam();

    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f - 5.0f;
    }

    double expected = 0.0;
    for (int i = 0; i < n; i++) {
        expected += h_input[i];
    }

    float *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    float result = vectorSum_GPU_Warp(d_input, n, THREADS_PER_BLOCK);

    double rel_error = std::fabs(result - expected) / std::fabs(expected);
    printf("[vectorSum_GPU_Warp @ %d]: rel_error = %.2e\n", n, rel_error);
    EXPECT_LT(rel_error, SUM_ERROR_THRESHOLD);

    cudaFree(d_input);
}

TEST_P(SumReduceTest, Threshold_CorrectnessAgainstCPU) {
    int n = GetParam();

    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f - 5.0f;
    }

    double expected = 0.0;
    for (int i = 0; i < n; i++) {
        expected += h_input[i];
    }

    float *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    float result = vectorSum_Threshold(d_input, n, THREADS_PER_BLOCK, 1000);

    double rel_error = std::fabs(result - expected) / std::fabs(expected);
    printf("[vectorSum_Threshold @ %d]: rel_error = %.2e\n", n, rel_error);
    EXPECT_LT(rel_error, SUM_ERROR_THRESHOLD);

    cudaFree(d_input);
}

TEST_P(SumReduceTest, Atomic_CorrectnessAgainstCPU) {
    int n = GetParam();

    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f - 5.0f;
    }

    double expected = 0.0;
    for (int i = 0; i < n; i++) {
        expected += h_input[i];
    }

    float *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    float result = vectorSum_Atomic(d_input, n, THREADS_PER_BLOCK);

    double rel_error = std::fabs(result - expected) / std::fabs(expected);
    printf("[vectorSum_Atomic @ %d]: rel_error = %.2e\n", n, rel_error);
    EXPECT_LT(rel_error, SUM_ERROR_THRESHOLD);

    cudaFree(d_input);
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    SumReduceTest,
    ::testing::ValuesIn(TEST_SIZES),
    [](const ::testing::TestParamInfo<int>& info) {
        return "size_" + std::to_string(info.param);
    }
);

// ============================================================================
// Max Reduction Tests
// ============================================================================

class MaxReduceTest : public ReduceTest,
                      public ::testing::WithParamInterface<int> {
};

TEST_P(MaxReduceTest, GPU_CorrectnessAgainstCPU) {
    int n = GetParam();

    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f - 5.0f;
    }

    // CPU reference
    float expected = h_input[0];
    for (int i = 1; i < n; i++) {
        if (h_input[i] > expected) {
            expected = h_input[i];
        }
    }

    float *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    float result = vectorMax_GPU(d_input, n, THREADS_PER_BLOCK);

    // Max should be exact (no accumulation)
    printf("[vectorMax_GPU @ %d]: result = %.6f, expected = %.6f\n", n, result, expected);
    EXPECT_FLOAT_EQ(result, expected);

    cudaFree(d_input);
}

TEST_P(MaxReduceTest, GPU_Warp_CorrectnessAgainstCPU) {
    int n = GetParam();

    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f - 5.0f;
    }

    float expected = h_input[0];
    for (int i = 1; i < n; i++) {
        if (h_input[i] > expected) {
            expected = h_input[i];
        }
    }

    float *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    float result = vectorMax_GPU_Warp(d_input, n, THREADS_PER_BLOCK);

    printf("[vectorMax_GPU_Warp @ %d]: result = %.6f, expected = %.6f\n", n, result, expected);
    EXPECT_FLOAT_EQ(result, expected);

    cudaFree(d_input);
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    MaxReduceTest,
    ::testing::ValuesIn(TEST_SIZES),
    [](const ::testing::TestParamInfo<int>& info) {
        return "size_" + std::to_string(info.param);
    }
);
