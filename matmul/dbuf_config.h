#ifndef DBUF_CONFIG_H
#define DBUF_CONFIG_H

#include <cstdlib>
#include <cstring>
#include <cstdio>

// Autotune config for warptile_dbuf. One global instance (g_dbuf) is set from
// CLI flags by parseDbufFlags(); the kernel dispatch reads it at launch time.
struct DbufConfig {
    int BM, BN, BK;      // block tile
    int WM, WN;          // warp tile
    int WNITER;          // warp-subtile iterations along N
    int TM, TN;          // thread tile
    int NUM_THREADS;     // threads per block
};

extern DbufConfig g_dbuf;

// Parse --dbuf-BM=128 style flags from argv; returns adjusted argc (flags
// consumed). Call before the benchmark parses its own options.
inline int parseDbufFlags(int argc, char *argv[]) {
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        int *dst = nullptr;
        if      (strncmp(a, "--dbuf-BM=", 10) == 0) dst = &g_dbuf.BM;
        else if (strncmp(a, "--dbuf-BN=", 10) == 0) dst = &g_dbuf.BN;
        else if (strncmp(a, "--dbuf-BK=", 10) == 0) dst = &g_dbuf.BK;
        else if (strncmp(a, "--dbuf-WM=", 10) == 0) dst = &g_dbuf.WM;
        else if (strncmp(a, "--dbuf-WN=", 10) == 0) dst = &g_dbuf.WN;
        else if (strncmp(a, "--dbuf-WNITER=", 14) == 0) dst = &g_dbuf.WNITER;
        else if (strncmp(a, "--dbuf-TM=", 10) == 0) dst = &g_dbuf.TM;
        else if (strncmp(a, "--dbuf-TN=", 10) == 0) dst = &g_dbuf.TN;
        else if (strncmp(a, "--dbuf-NT=", 10) == 0) dst = &g_dbuf.NUM_THREADS;
        if (dst != nullptr) {
            *dst = atoi(a + (strncmp(a, "--dbuf-WNITER=", 14) == 0 ? 14 : 10));
            continue;
        }
        argv[out++] = argv[i];
    }
    argv[out] = nullptr;
    return out;
}

inline void printDbufConfig(const char *tag) {
    fprintf(stderr, "[dbuf] %s BM=%d BN=%d BK=%d WM=%d WN=%d WNITER=%d TM=%d TN=%d NT=%d\n",
            tag, g_dbuf.BM, g_dbuf.BN, g_dbuf.BK, g_dbuf.WM, g_dbuf.WN,
            g_dbuf.WNITER, g_dbuf.TM, g_dbuf.TN, g_dbuf.NUM_THREADS);
}

// ---------------------------------------------------------------------------
// In-process autotuner (like @triton.autotune): sweep all configs from
// dbuf_configs.inc in ONE CUDA session — matrices allocated once, kernel
// launched per config with cuda-event timing. Prints one CSV row per config
// (flushed) and finally the best config + TFLOPS.
// Implemented in matmul_warptile_dbuf.cu (has access to the kernel launcher).
// ---------------------------------------------------------------------------
void dbufAutotune(int N, int num_iterations);

#endif // DBUF_CONFIG_H
