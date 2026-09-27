#!/bin/bash
set -e
NVCC=/usr/local/cuda/bin/nvcc
cd "$(dirname "$0")/.."
$NVCC -O3 -arch=sm_90 -std=c++17 \
  matmul/single_warp16.cu \
  cuda_utils.cu \
  -I . -I matmul \
  -lnvidia-ml \
  -o single_warp16
echo "BUILD OK: $(stat -c%s single_warp16) bytes"
