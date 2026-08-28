#!/bin/bash
set -e
cd /tmp/DeepGEMM
rm -rf build *.egg-info
export CPLUS_INCLUDE_PATH=/opt/conda/lib/python3.11/site-packages/cutlass_library/source/include:$CPLUS_INCLUDE_PATH
export TORCH_CUDA_ARCH_LIST="9.0"
pip install -e . --no-build-isolation -q 2>&1 | grep -iE "error|Successfully" | head -5 || true
python3 -c "import deep_gemm; print('deep_gemm import OK')" 2>&1 | tail -3
