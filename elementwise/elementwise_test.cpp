#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "vector_add.h"
#include "vector_mul.h"
#include "vector_fma.h"

// ============================================================================
// Test Configuration
// ============================================================================

const int THREADS_PER_BLOCK = 256;
const std::vector<int> TEST_SIZES = {16, 1024, 65536, 1048576};
const double ERROR_THRESHOLD = 1e-6;

// ============================================================================
// Test Fixture
// ============================================================================

class ElementwiseTest : public ::testing::Test {
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
// Vector Add Tests
// ============================================================================

class VectorAddTest : public ElementwiseTest,
                      public ::testing::WithParamInterface<int> {
};

TEST_P(VectorAddTest, CorrectnessAgainstCPU) {
    int n = GetParam();

    // Allocate host memory
    std::vector<float> h_a(n), h_b(n), h_c(n), h_expected(n);

    // Initialize with deterministic values
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f;
        h_b[i] = static_cast<float>((i * 11 + 17) % 1000) / 100.0f;
        h_expected[i] = h_a[i] + h_b[i];
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, n * sizeof(float)), cudaSuccess);

    // Copy to device
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    // Launch kernel
    int blocksPerGrid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaDeviceSynchronize();

    // Copy result back
    ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify
    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        double error = std::fabs(h_c[i] - h_expected[i]);
        max_error = std::max(max_error, error);
    }

    printf("[vectorAdd @ %d]: max_error = %.2e\n", n, max_error);
    EXPECT_LT(max_error, ERROR_THRESHOLD);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    VectorAddTest,
    ::testing::ValuesIn(TEST_SIZES),
    [](const ::testing::TestParamInfo<int>& info) {
        return "size_" + std::to_string(info.param);
    }
);

// ============================================================================
// Vector Mul Tests
// ============================================================================

class VectorMulTest : public ElementwiseTest,
                      public ::testing::WithParamInterface<int> {
};

TEST_P(VectorMulTest, CorrectnessAgainstCPU) {
    int n = GetParam();

    std::vector<float> h_a(n), h_b(n), h_c(n), h_expected(n);

    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f;
        h_b[i] = static_cast<float>((i * 11 + 17) % 1000) / 100.0f;
        h_expected[i] = h_a[i] * h_b[i];
    }

    float *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, n * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    int blocksPerGrid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorMul<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaDeviceSynchronize();

    ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        double error = std::fabs(h_c[i] - h_expected[i]);
        max_error = std::max(max_error, error);
    }

    printf("[vectorMul @ %d]: max_error = %.2e\n", n, max_error);
    EXPECT_LT(max_error, ERROR_THRESHOLD);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    VectorMulTest,
    ::testing::ValuesIn(TEST_SIZES),
    [](const ::testing::TestParamInfo<int>& info) {
        return "size_" + std::to_string(info.param);
    }
);

// ============================================================================
// Vector FMA Tests
// ============================================================================

class VectorFMATest : public ElementwiseTest,
                      public ::testing::WithParamInterface<int> {
};

TEST_P(VectorFMATest, CorrectnessAgainstCPU) {
    int n = GetParam();

    std::vector<float> h_a(n), h_b(n), h_c(n), h_d(n), h_expected(n);

    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f;
        h_b[i] = static_cast<float>((i * 11 + 17) % 1000) / 100.0f;
        h_c[i] = static_cast<float>((i * 13 + 19) % 1000) / 100.0f;
        h_expected[i] = h_a[i] * h_b[i] + h_c[i];
    }

    float *d_a, *d_b, *d_c, *d_d;
    ASSERT_EQ(cudaMalloc(&d_a, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_d, n * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_c, h_c.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    int blocksPerGrid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorFMA<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, d_d, n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaDeviceSynchronize();

    ASSERT_EQ(cudaMemcpy(h_d.data(), d_d, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        double error = std::fabs(h_d[i] - h_expected[i]);
        max_error = std::max(max_error, error);
    }

    printf("[vectorFMA @ %d]: max_error = %.2e\n", n, max_error);
    EXPECT_LT(max_error, ERROR_THRESHOLD);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    VectorFMATest,
    ::testing::ValuesIn(TEST_SIZES),
    [](const ::testing::TestParamInfo<int>& info) {
        return "size_" + std::to_string(info.param);
    }
);

// ============================================================================
// Vector FMA Float4 Tests
// ============================================================================

class VectorFMAFloat4Test : public ElementwiseTest,
                            public ::testing::WithParamInterface<int> {
};

TEST_P(VectorFMAFloat4Test, CorrectnessAgainstCPU) {
    int n = GetParam();

    // float4 requires n to be multiple of 4
    if (n % 4 != 0) {
        GTEST_SKIP() << "Size must be multiple of 4 for float4 kernel";
    }

    std::vector<float> h_a(n), h_b(n), h_c(n), h_d(n), h_expected(n);

    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>((i * 7 + 13) % 1000) / 100.0f;
        h_b[i] = static_cast<float>((i * 11 + 17) % 1000) / 100.0f;
        h_c[i] = static_cast<float>((i * 13 + 19) % 1000) / 100.0f;
        h_expected[i] = h_a[i] * h_b[i] + h_c[i];
    }

    float *d_a, *d_b, *d_c, *d_d;
    ASSERT_EQ(cudaMalloc(&d_a, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_d, n * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_c, h_c.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    int n_vec = n / 4;
    int blocksPerGrid = (n_vec + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorFMA_float4<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, d_d, n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaDeviceSynchronize();

    ASSERT_EQ(cudaMemcpy(h_d.data(), d_d, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        double error = std::fabs(h_d[i] - h_expected[i]);
        max_error = std::max(max_error, error);
    }

    printf("[vectorFMA_float4 @ %d]: max_error = %.2e\n", n, max_error);
    EXPECT_LT(max_error, ERROR_THRESHOLD);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    VectorFMAFloat4Test,
    ::testing::ValuesIn(TEST_SIZES),
    [](const ::testing::TestParamInfo<int>& info) {
        return "size_" + std::to_string(info.param);
    }
);
