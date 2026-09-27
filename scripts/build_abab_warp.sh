#!/bin/bash
set -e
NVCC=/usr/local/cuda/bin/nvcc
cd "$(dirname "$0")/.."
$NVCC -O3 -arch=sm_90 -std=c++17 \
  matmul/abab_warp.cu \
  matmul/matmul_warptile_tuned.cu \
  cuda_utils.cu \
  -I . -I matmul \
  -lnvidia-ml \
  -o abab_warp
echo "BUILD OK: $(stat -c%s abab_warp) bytes"
