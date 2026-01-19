#ifndef CUDA_COMMON_BENCHMARK_UTILS_H
#define CUDA_COMMON_BENCHMARK_UTILS_H

#include <string>
#include <vector>
#include <functional>

// ============================================================================
// Benchmark Result Structures
// ============================================================================

// Per-iteration timing statistics (percentiles)
struct TimingStats {
    float p50_ms;  // Median (50th percentile)
    float p90_ms;  // 90th percentile
};

// Result of a single benchmark run
struct BenchmarkResult {
    float p50_time_ms;       // Median (50th percentile), -1.0 if skipped/failed
    float p90_time_ms;       // 90th percentile, -1.0 if skipped/failed
    bool skipped;
    std::string skip_reason;

    // Default constructor - marks as skipped
    BenchmarkResult()
        : p50_time_ms(-1.0f), p90_time_ms(-1.0f), skipped(true), skip_reason("Not run") {}

    // Constructor for successful result
    BenchmarkResult(float p50, float p90)
        : p50_time_ms(p50), p90_time_ms(p90), skipped(false), skip_reason("") {}

    // Constructor for skipped/failed result
    BenchmarkResult(const std::string& reason)
        : p50_time_ms(-1.0f), p90_time_ms(-1.0f), skipped(true), skip_reason(reason) {}
};

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig {
    int warmup_iterations = 10;
    int timed_iterations = 100;
};

// ============================================================================
// Timing Functions
// ============================================================================

// Get per-iteration timing statistics using CUDA events.
// Uses per-iteration event pairs without CPU sync between iterations,
// then extracts P50 (median) and P90 percentiles for accurate kernel timing.
//
// Parameters:
//   kernel_fn: Callable that executes the kernel once (no arguments)
//   config: Benchmark configuration (warmup and timed iterations)
//
// Returns:
//   TimingStats with p50_ms and p90_ms values
//
// Example:
//   auto kernel_fn = [&]() { kernel->execute(d_input, d_output); };
//   TimingStats stats = get_timing_stats(kernel_fn, config);
TimingStats get_timing_stats(const std::function<void()>& kernel_fn, const BenchmarkConfig& config);

// ============================================================================
// Table Printing Utilities
// ============================================================================

// Print benchmark table header
// Parameters:
//   title: Table title (e.g., "SOFTMAX PERFORMANCE BENCHMARK")
//   size_labels: Column headers for each size (e.g., {"16", "1K", "1M"})
//   iterations: Number of iterations per test
void print_benchmark_header(const char* title, const std::vector<std::string>& size_labels, int iterations);

// Print a single benchmark row
// Parameters:
//   method_name: Name of the method being benchmarked
//   results: Benchmark results for each size
void print_benchmark_row(const char* method_name, const std::vector<BenchmarkResult>& results);

// Print benchmark table footer with skip reasons
// Parameters:
//   method_names: Names of all methods
//   all_results: 2D array of results [method][size]
void print_benchmark_footer(const std::vector<std::string>& method_names,
                            const std::vector<std::vector<BenchmarkResult>>& all_results);

// Print top N fastest methods per size
// Parameters:
//   method_names: Names of all methods
//   all_results: 2D array of results [method][size]
//   size_labels: Labels for each size column
//   top_n: Number of top methods to show (default 3)
//   exclude_methods: Method names to exclude from ranking (e.g., {"naive"})
void print_top_fastest(const std::vector<std::string>& method_names,
                       const std::vector<std::vector<BenchmarkResult>>& all_results,
                       const std::vector<std::string>& size_labels,
                       int top_n = 3,
                       const std::vector<std::string>& exclude_methods = {});

#endif  // CUDA_COMMON_BENCHMARK_UTILS_H
