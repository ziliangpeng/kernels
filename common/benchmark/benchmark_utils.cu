#include "benchmark_utils.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>

// ============================================================================
// Timing Implementation
// ============================================================================

TimingStats get_timing_stats(const std::function<void()>& kernel_fn, const BenchmarkConfig& config) {
    int num_iterations = config.timed_iterations;

    // Allocate event pairs for each iteration
    std::vector<cudaEvent_t> starts(num_iterations), stops(num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        cudaEventCreate(&starts[i]);
        cudaEventCreate(&stops[i]);
    }

    // Warmup: run a few iterations to ensure kernel is compiled and caches are warm
    for (int i = 0; i < config.warmup_iterations; i++) {
        kernel_fn();
    }
    cudaDeviceSynchronize();  // No events recorded yet, use device sync

    // Queue all iterations with per-iteration events (no sync between)
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(starts[i]);
        kernel_fn();
        cudaEventRecord(stops[i]);
    }

    // Single sync at the end - wait for last event
    cudaEventSynchronize(stops[num_iterations - 1]);

    // Extract individual timings
    std::vector<float> timings(num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        cudaEventElapsedTime(&timings[i], starts[i], stops[i]);
    }

    // Sort for percentile calculation
    std::sort(timings.begin(), timings.end());

    TimingStats stats;
    stats.p50_ms = timings[num_iterations / 2];            // 50th percentile (median)
    stats.p90_ms = timings[(num_iterations * 90) / 100];   // 90th percentile

    // Cleanup
    for (int i = 0; i < num_iterations; i++) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }

    return stats;
}

// ============================================================================
// Table Printing Implementation
// ============================================================================

void print_benchmark_header(const char* title, const std::vector<std::string>& size_labels, int iterations) {
    printf("\n");
    printf("=======================================================================================\n");
    printf("                         %s\n", title);
    printf("=======================================================================================\n");
    printf("Iterations per test: %d\n", iterations);
    printf("Metric: P50/P90 execution time (ms) - median and 90th percentile\n");
    printf("Method: Per-iteration CUDA event timing (no CPU sync between iterations)\n\n");

    // Print header row
    printf("%-15s", "Method");
    for (size_t s = 0; s < size_labels.size(); s++) {
        printf(" | %-13s", size_labels[s].c_str());
    }
    printf(" |\n");

    // Print separator
    printf("%-15s", "---------------");
    for (size_t s = 0; s < size_labels.size(); s++) {
        printf("-|--------------");
    }
    printf("-|\n");
}

void print_benchmark_row(const char* method_name, const std::vector<BenchmarkResult>& results) {
    printf("%-15s", method_name);
    for (size_t s = 0; s < results.size(); s++) {
        if (results[s].skipped) {
            printf(" | %-13s", "SKIPPED");
        } else if (results[s].p50_time_ms < 0) {
            printf(" | %-13s", "FAILED");
        } else {
            printf(" | %5.3f/%5.3f", results[s].p50_time_ms, results[s].p90_time_ms);
        }
    }
    printf(" |\n");
}

void print_benchmark_footer(const std::vector<std::string>& method_names,
                            const std::vector<std::vector<BenchmarkResult>>& all_results) {
    // Print skipped reasons
    printf("\nSKIPPED/FAILED reasons:\n");
    for (size_t m = 0; m < method_names.size(); m++) {
        bool has_skip = false;
        for (size_t s = 0; s < all_results[m].size(); s++) {
            if (all_results[m][s].skipped || all_results[m][s].p50_time_ms < 0) {
                if (!has_skip) {
                    printf("- %s: %s\n", method_names[m].c_str(), all_results[m][s].skip_reason.c_str());
                    has_skip = true;
                }
            }
        }
    }
}

void print_top_fastest(const std::vector<std::string>& method_names,
                       const std::vector<std::vector<BenchmarkResult>>& all_results,
                       const std::vector<std::string>& size_labels,
                       int top_n,
                       const std::vector<std::string>& exclude_methods) {
    printf("\nTOP %d FASTEST per size by P50", top_n);
    if (!exclude_methods.empty()) {
        printf(" (excluding");
        for (size_t i = 0; i < exclude_methods.size(); i++) {
            printf(" %s", exclude_methods[i].c_str());
            if (i < exclude_methods.size() - 1) printf(",");
        }
        printf(")");
    }
    printf(":\n");

    size_t num_sizes = size_labels.size();

    for (size_t s = 0; s < num_sizes; s++) {
        // Create array of (method_index, time) pairs
        struct MethodTime {
            size_t method_idx;
            float time;
        };
        std::vector<MethodTime> method_times;

        // Collect valid methods
        for (size_t m = 0; m < method_names.size(); m++) {
            // Skip excluded methods
            bool is_excluded = false;
            for (const auto& excluded : exclude_methods) {
                if (method_names[m] == excluded) {
                    is_excluded = true;
                    break;
                }
            }
            if (is_excluded) continue;

            if (!all_results[m][s].skipped && all_results[m][s].p50_time_ms > 0) {
                method_times.push_back({m, all_results[m][s].p50_time_ms});
            }
        }

        // Sort by time
        std::sort(method_times.begin(), method_times.end(),
                  [](const MethodTime& a, const MethodTime& b) { return a.time < b.time; });

        // Print top N
        printf("- %-5s: ", size_labels[s].c_str());
        size_t print_count = std::min(static_cast<size_t>(top_n), method_times.size());
        for (size_t i = 0; i < print_count; i++) {
            if (i > 0) printf(", ");
            printf("%s (%.3f ms)", method_names[method_times[i].method_idx].c_str(),
                   method_times[i].time);
        }
        printf("\n");
    }
    printf("=======================================================================================\n");
}
