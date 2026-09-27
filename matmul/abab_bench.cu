// abab_bench.cu — Rigorous A/B benchmark for the tuned-vs-typed FP16 puzzle.
//
// What Ziliang asked for, implemented:
//   * same process, same GPU, alternating A-B-A-B (order effects and thermal
//     drift hit both equally)
//   * CUDA-event timing, exactly like his recipe: record(start) → kernel →
//     record(stop) → elapsed
//   * TWO timing modes per measurement:
//       batched : one event pair around 100 back-to-back launches (the
//                 existing harness style; hides launch overhead)
//       single  : one event pair per launch, median of 100 (classic recipe;
//                 includes any inter-launch gap that lands on the GPU)
//   * N>=5 repetitions, report median + IQR (no "within noise" without data)
//   * SM clock sampled after each phase via NVML
//   * verify both paths produce identical C before timing
//
// A = typed, hard-coded FP16 kernel   (method 1d_blocktile_f16, config 64,64,8,8)
// B = tuned template dispatch FP16 kernel (method 1d_autotune_f16, same config)
//
// Question this answers: is B's runtime dispatch + template codegen measurably
// slower than A's hard-coded kernel, when both are timed on the same GPU in
// the same process with clean CUDA events?

#include "matmul_kernel.h"
#include "matmul_1d_blocktile_typed.h"
#include "matmul_1d_blocktile_tuned.h"
#include "dtype_traits.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>

static double median(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    return n % 2 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}
static double iqr(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    double q1 = v[n/4], q3 = v[(3*n)/4];
    return q3 - q1;
}

static unsigned sm_clock() {
    nvmlDevice_t dev; unsigned clock = 0;
    if (nvmlInit() == NVML_SUCCESS &&
        nvmlDeviceGetHandleByIndex(0, &dev) == NVML_SUCCESS) {
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clock);
    }
    return clock;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 4096;
    int reps = argc > 2 ? atoi(argv[2]) : 7;

    cudaFree(0);  // establish context

    // Host data (deterministic)
    float *h_A = (float*)malloc((size_t)N*N*sizeof(float));
    float *h_B = (float*)malloc((size_t)N*N*sizeof(float));
    for (long long i = 0; i < (long long)N*N; i++) {
        h_A[i] = ((i * 1103515245 + 12345) % 1000) / 1000.0f - 0.5f;
        h_B[i] = ((i * 2654435761 + 987654321) % 1000) / 1000.0f - 0.5f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)N*N*sizeof(float));
    cudaMalloc(&d_B, (size_t)N*N*sizeof(float));
    cudaMalloc(&d_C, (size_t)N*N*sizeof(float));
    cudaMemcpy(d_A, h_A, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice);

    // A: typed hard-coded kernel, default config (64,64,8,8)
    MatmulKernel *A = new Matmul1DBlocktileTyped<DTypeTraitsHalf>(N, 0);
    // B: tuned template kernel, same config installed
    MatmulKernel *B = new Matmul1DBlocktileTuned<DTypeTraitsHalf>(N, 0, Cfg1D{64, 64, 8, 8});

    // --- verify A and B agree before timing anything ---
    A->execute(d_A, d_B, d_C); cudaDeviceSynchronize();
    std::vector<float> cA((size_t)N*N);
    cudaMemcpy(cA.data(), d_C, (size_t)N*N*sizeof(float), cudaMemcpyDeviceToHost);
    B->execute(d_A, d_B, d_C); cudaDeviceSynchronize();
    std::vector<float> cB((size_t)N*N);
    cudaMemcpy(cB.data(), d_C, (size_t)N*N*sizeof(float), cudaMemcpyDeviceToHost);
    double maxdiff = 0.0; 
    for (size_t i = 0; i < cA.size(); i++) maxdiff = std::max(maxdiff, (double)std::fabs(cA[i]-cB[i]));
    printf("# verify A-vs-B maxdiff = %.3e  %s\n", maxdiff, maxdiff < 1e-6 ? "IDENTICAL" : "DIFFER");

    // --- timing helpers (both CUDA-event styles) ---
    cudaEvent_t es, ee;
    cudaEventCreate(&es); cudaEventCreate(&ee);

    auto time_batched = [&](MatmulKernel *k, int iters) {
        for (int w = 0; w < 10; w++) k->execute(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        cudaEventRecord(es);
        for (int i = 0; i < iters; i++) k->execute(d_A, d_B, d_C);
        cudaEventRecord(ee); cudaEventSynchronize(ee);
        float ms; cudaEventElapsedTime(&ms, es, ee);
        return ms / iters;
    };
    auto time_single = [&](MatmulKernel *k, int iters) {
        for (int w = 0; w < 10; w++) k->execute(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        std::vector<double> per;
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(es);
            k->execute(d_A, d_B, d_C);
            cudaEventRecord(ee); cudaEventSynchronize(ee);
            float ms; cudaEventElapsedTime(&ms, es, ee);
            per.push_back(ms);
        }
        return median(per);
    };

    // --- ABAB x reps ---
    struct Res { double batched[32], single[32]; int n; } rA{}, rB{};
    printf("# node=%s GPU=%s\n", getenv("HOSTNAME") ? getenv("HOSTNAME") : "?",
           "H100");
    printf("# rep,A_batched_ms,A_single_ms,B_batched_ms,B_single_ms,sm_clock\n");
    for (int r = 0; r < reps; r++) {
        rA.batched[r] = time_batched(A, 100);
        rA.single[r]  = time_single(A, 100);
        rB.batched[r] = time_batched(B, 100);
        rB.single[r]  = time_single(B, 100);
        unsigned clk = sm_clock();
        printf("R,%d,%.4f,%.4f,%.4f,%.4f,%u\n", r,
               rA.batched[r], rA.single[r], rB.batched[r], rB.single[r], clk);
        fflush(stdout);
    }

    std::vector<double> vA_b(rA.batched, rA.batched + reps), vB_b(rB.batched, rB.batched + reps);
    std::vector<double> vA_s(rA.single, rA.single + reps), vB_s(rB.single, rB.single + reps);
    double ab = median(vA_b), bb = median(vB_b), as = median(vA_s), bs = median(vB_s);
    printf("# SUMMARY (median of %d reps)\n", reps);
    printf("# A typed   batched %.4f ms  (%.2f TFLOPS)  single %.4f ms (%.2f TFLOPS)\n",
           ab, 2.0*N*N*N/ab/1e9, as, 2.0*N*N*N/as/1e9);
    printf("# B tuned   batched %.4f ms  (%.2f TFLOPS)  single %.4f ms (%.2f TFLOPS)\n",
           bb, 2.0*N*N*N/bb/1e9, bs, 2.0*N*N*N/bs/1e9);
    printf("# B/A batched = %.4f  (IQR A %.5f, B %.5f)\n", bb/ab, iqr(vA_b), iqr(vB_b));
    printf("# B/A single  = %.4f  (IQR A %.5f, B %.5f)\n", bs/as, iqr(vA_s), iqr(vB_s));

    // --- launch-overhead probe: empty-kernel dispatch cost of B's if-chain ---
    // (host-side; batched timing hides it only if host stays ahead of GPU)
    {
        cudaEventRecord(es);
        for (int i = 0; i < 100; i++) B->execute(d_A, d_B, d_C);
        cudaEventRecord(ee); cudaEventSynchronize(ee);
        float ms; cudaEventElapsedTime(&ms, es, ee);
        // CPU wall time for the same launches:
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) B->execute(d_A, d_B, d_C);
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();
        double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;
        printf("# B launch: GPU-side %.1f us/launch, host-side %.1f us/launch %s\n",
               ms * 10.0, wall_us,
               wall_us * 1000.0 > ms * 10.0 * 1000.0 ? "(HOST-BOUND!)" : "(gpu-bound)");
    }

    delete A; delete B;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return 0;
}
