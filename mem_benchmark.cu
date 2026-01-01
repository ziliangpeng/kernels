#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

// Detect PCIe generation and return theoretical bandwidth
float get_pcie_bandwidth() {
    FILE *fp = popen("nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader -i 0", "r");
    if (!fp) {
        fprintf(stderr, "Warning: Could not detect PCIe config, assuming Gen4 x16\n");
        return 32.0f;
    }

    int gen, width;
    if (fscanf(fp, "%d, %d", &gen, &width) != 2) {
        pclose(fp);
        fprintf(stderr, "Warning: Could not parse PCIe config, assuming Gen4 x16\n");
        return 32.0f;
    }
    pclose(fp);

    // Calculate bandwidth per lane (GB/s)
    // GT/s = GigaTransfers/sec (billions of clock cycles per second, 1 bit per transfer per lane)
    // Formula: Transfer_Rate (GT/s) × Encoding_Efficiency / 8
    // Encoding: 128b/130b for Gen3+ = 0.9846 efficiency (2 bits overhead per 128 data bits)
    float bandwidth_per_lane;
    switch (gen) {
        case 3: bandwidth_per_lane = 0.985f; break;  // 8 GT/s × 0.9846 / 8
        case 4: bandwidth_per_lane = 1.969f; break;  // 16 GT/s × 0.9846 / 8
        case 5: bandwidth_per_lane = 3.938f; break;  // 32 GT/s × 0.9846 / 8
        case 6: bandwidth_per_lane = 7.877f; break;  // 64 GT/s × 0.9846 / 8
        default:
            fprintf(stderr, "Warning: Unknown PCIe gen %d, assuming Gen4\n", gen);
            bandwidth_per_lane = 1.969f;
    }

    return bandwidth_per_lane * width;
}

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N       Set array size in MB (default: 100)\n");
    printf("  -p, --pinned       Use pinned memory (default: pageable)\n");
    printf("  -i, --iterations N Number of iterations (default: 1)\n");
    printf("  -h, --help         Show this help message\n");
}

int main(int argc, char *argv[]) {
    // Default parameters
    int size_mb = 100;
    bool use_pinned = false;
    int iterations = 1;

    // Parse command line arguments
    static struct option long_options[] = {
        {"size",       required_argument, 0, 'n'},
        {"pinned",     no_argument,       0, 'p'},
        {"iterations", required_argument, 0, 'i'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:pi:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                size_mb = atoi(optarg);
                if (size_mb <= 0) {
                    fprintf(stderr, "Error: size must be positive\n");
                    return 1;
                }
                break;
            case 'p':
                use_pinned = true;
                break;
            case 'i':
                iterations = atoi(optarg);
                if (iterations <= 0) {
                    fprintf(stderr, "Error: iterations must be positive\n");
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
        }
    }

    // Calculate size in bytes
    size_t bytes = (size_t)size_mb * 1024 * 1024;

    // Detect PCIe bandwidth
    float pcie_bandwidth_gbs = get_pcie_bandwidth();

    printf("===========================================\n");
    printf("Memory Transfer Benchmark\n");
    printf("===========================================\n");
    printf("Size: %d MB (%zu bytes)\n", size_mb, bytes);
    printf("Memory type: %s\n", use_pinned ? "PINNED" : "PAGEABLE");
    printf("Iterations: %d\n", iterations);
    printf("PCIe: %.1f GB/s theoretical (detected)\n", pcie_bandwidth_gbs);
    printf("===========================================\n\n");

    // Measure allocation time
    float *h_data;
    cudaEvent_t alloc_start, alloc_stop;
    cudaCheckError(cudaEventCreate(&alloc_start));
    cudaCheckError(cudaEventCreate(&alloc_stop));

    printf("Allocating host memory...\n");
    cudaEventRecord(alloc_start);
    if (use_pinned) {
        // cudaMallocHost = malloc + mlock (pin pages) + cudaHostRegister (DMA mapping).
        // Plain malloc + mlock alone won't enable fast DMA path without registration.
        cudaCheckError(cudaMallocHost(&h_data, bytes));
    } else {
        h_data = (float*)malloc(bytes);
        if (!h_data) {
            fprintf(stderr, "Failed to allocate host memory\n");
            return 1;
        }
    }
    cudaEventRecord(alloc_stop);
    cudaEventSynchronize(alloc_stop);

    float alloc_time_ms;
    cudaEventElapsedTime(&alloc_time_ms, alloc_start, alloc_stop);
    printf("  Allocation time: %.3f ms\n\n", alloc_time_ms);

    // Initialize data
    printf("Initializing data...\n");
    for (size_t i = 0; i < bytes / sizeof(float); i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data;
    cudaCheckError(cudaMalloc(&d_data, bytes));

    // Benchmark Host -> Device transfers
    printf("\n--- Host -> Device Transfer ---\n");
    float h2d_total = 0;

    // Create events once outside loop to avoid overhead
    cudaEvent_t h2d_start, h2d_stop;
    cudaCheckError(cudaEventCreate(&h2d_start));
    cudaCheckError(cudaEventCreate(&h2d_stop));

    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(h2d_start);
        cudaCheckError(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        cudaEventRecord(h2d_stop);
        cudaEventSynchronize(h2d_stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, h2d_start, h2d_stop);
        h2d_total += time_ms;

        float bandwidth_gbs = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
        printf("  Iter %2d: %.3f ms -> %.2f GB/s\n", i + 1, time_ms, bandwidth_gbs);
    }

    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    float h2d_avg = h2d_total / iterations;
    float h2d_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (h2d_avg / 1000.0f);
    printf("  Average: %.3f ms -> %.2f GB/s\n", h2d_avg, h2d_bandwidth);

    // Benchmark Device -> Host transfers
    printf("\n--- Device -> Host Transfer ---\n");
    float d2h_total = 0;

    // Create events once outside loop to avoid overhead
    cudaEvent_t d2h_start, d2h_stop;
    cudaCheckError(cudaEventCreate(&d2h_start));
    cudaCheckError(cudaEventCreate(&d2h_stop));

    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(d2h_start);
        cudaCheckError(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        cudaEventRecord(d2h_stop);
        cudaEventSynchronize(d2h_stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, d2h_start, d2h_stop);
        d2h_total += time_ms;

        float bandwidth_gbs = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
        printf("  Iter %2d: %.3f ms -> %.2f GB/s\n", i + 1, time_ms, bandwidth_gbs);
    }

    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
    float d2h_avg = d2h_total / iterations;
    float d2h_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (d2h_avg / 1000.0f);
    printf("  Average: %.3f ms -> %.2f GB/s\n", d2h_avg, d2h_bandwidth);

    // Summary
    float h2d_efficiency = (h2d_bandwidth / pcie_bandwidth_gbs) * 100.0f;
    float d2h_efficiency = (d2h_bandwidth / pcie_bandwidth_gbs) * 100.0f;

    printf("\n===========================================\n");
    printf("SUMMARY\n");
    printf("===========================================\n");
    printf("Memory type:     %s\n", use_pinned ? "PINNED" : "PAGEABLE");
    printf("Allocation time: %.3f ms\n", alloc_time_ms);
    printf("H->D bandwidth:  %.2f GB/s (%.1f%% of peak, avg over %d iters)\n",
           h2d_bandwidth, h2d_efficiency, iterations);
    printf("D->H bandwidth:  %.2f GB/s (%.1f%% of peak, avg over %d iters)\n",
           d2h_bandwidth, d2h_efficiency, iterations);
    printf("===========================================\n");

    // Cleanup
    if (use_pinned) {
        cudaFreeHost(h_data);
    } else {
        free(h_data);
    }
    cudaFree(d_data);
    cudaEventDestroy(alloc_start);
    cudaEventDestroy(alloc_stop);

    return 0;
}
