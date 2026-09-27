#!/bin/bash
# Build abab_vec (hard-coded FP16 vec winner vs dispatch path, ABAB rigor).
set -e
NVCC=/usr/local/cuda/bin/nvcc
cd "$(dirname "$0")/.."
mkdir -p .build
$NVCC -O3 -arch=sm_90 -std=c++17 \
  matmul/abab_vec.cu \
  matmul/matmul_vectorized_tuned.cu \
  cuda_utils.cu \
  -I . -I matmul \
  -lnvidia-ml \
  -o abab_vec
echo "BUILD OK: $(stat -c%s abab_vec) bytes"
