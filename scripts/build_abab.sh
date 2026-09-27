#!/bin/bash
# Build abab_bench (ABAB A/B rigor tool) — nvcc direct, sm_90, no -dc.
set -e
# CUDA module (compute nodes have nvcc via module)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
NVCC=/usr/local/cuda/bin/nvcc
command -v $NVCC >/dev/null || { echo "nvcc not found"; exit 1; }
cd "$(dirname "$0")/.."  # repo root
mkdir -p .build
$NVCC -O3 -arch=sm_90 -std=c++17 \
  matmul/abab_bench.cu \
  matmul/matmul_1d_blocktile_typed.cu \
  matmul/matmul_1d_blocktile_tuned.cu \
  cuda_utils.cu \
  -I . -I matmul \
  -Xlinker -L/usr/lib/x86_64-linux-gnu -lnvidia-ml \
  -o abab_bench
echo "BUILD OK: $(stat -c%s abab_bench) bytes"
