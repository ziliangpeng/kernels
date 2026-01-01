#include "cuda_utils.h"
#include <stdlib.h>
#include <stdio.h>

void calculate_and_print_statistics(float *timings, int num_iterations) {
    // Calculate statistics
    float min_time = timings[0];
    float max_time = timings[0];
    float sum_time = 0.0f;

    for (int i = 0; i < num_iterations; i++) {
        if (timings[i] < min_time) min_time = timings[i];
        if (timings[i] > max_time) max_time = timings[i];
        sum_time += timings[i];
    }
    float avg_time = sum_time / num_iterations;

    // Sort for percentiles
    qsort(timings, num_iterations, sizeof(float), [](const void *a, const void *b) {
        float fa = *(const float*)a;
        float fb = *(const float*)b;
        return (fa > fb) - (fa < fb);
    });

    // Calculate percentiles
    int p50_idx = (int)(num_iterations * 0.50);
    int p90_idx = (int)(num_iterations * 0.90);
    int p95_idx = (int)(num_iterations * 0.95);
    int p99_idx = (int)(num_iterations * 0.99);

    printf("\n===========================================\n");
    printf("Kernel Execution Statistics (%d runs):\n", num_iterations);
    printf("===========================================\n");
    printf("  Min:    %.3f ms\n", min_time);
    printf("  Max:    %.3f ms\n", max_time);
    printf("  Mean:   %.3f ms\n", avg_time);
    printf("  Median: %.3f ms\n", timings[p50_idx]);
    printf("  P90:    %.3f ms\n", timings[p90_idx]);
    printf("  P95:    %.3f ms\n", timings[p95_idx]);
    printf("  P99:    %.3f ms\n", timings[p99_idx]);
    printf("===========================================\n");

    free(timings);
}
