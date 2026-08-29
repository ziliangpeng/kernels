#!/bin/bash
# E2E A/B v2: warm up JIT with mixed shapes before benching; steady-state measurement;
# two workloads: short (512/256) and long-ctx (4k/256, 32k kv) to expose attention share.
set -u
MODEL=Qwen/Qwen3-1.7B
PORT=18888

launch() {
  local backend=$1 tag=$2
  local extra=""
  if [ "$backend" = "baseline" ]; then
    extra="--attention-backend TRITON_ATTN"
  else
    extra="--attention-backend ROCM_AITER_UNIFIED_ATTN"
  fi
  echo "[$(date +%T)] launching $backend"
  VLLM_ROCM_USE_AITER=1 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL --port $PORT --dtype bfloat16 \
    --max-model-len 40960 --gpu-memory-utilization 0.85 \
    $extra > /tmp/sv_$tag.log 2>&1 &
  for i in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || true)
    if [ "$code" = "200" ]; then echo "[$(date +%T)] $backend ready"; return 0; fi
    sleep 5
  done
  echo "TIMEOUT $backend"; return 1
}

# warm mixed shapes: short, medium, long prompts, several sizes to cover JIT variants
warm() {
  echo "  warming JIT (mixed shapes)..."
  for lens in "128 128" "512 256" "1024 256" "2048 128" "4096 256" "8192 256" "16384 256"; do
    set -- $lens
    python3 - << PYEOF > /dev/null 2>&1
import requests, concurrent.futures
def hit(i):
    body = {"model": "$MODEL", "messages": [{"role": "user", "content": "hi " * $1}], "max_tokens": $2, "temperature": 0}
    requests.post("http://localhost:$PORT/v1/chat/completions", json=body, timeout=600)
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    ex.map(hit, range(8))
PYEOF
  done
  echo "  warm done"
}

bench() {
  local in=$1 out=$2 tag=$3
  vllm bench serve \
    --backend openai-chat --model $MODEL \
    --base-url http://localhost:$PORT \
    --endpoint /v1/chat/completions \
    --dataset-name random --random-input-len $in --random-output-len $out \
    --num-prompts 96 --max-concurrency 32 \
    --save-result --result-dir /tmp --result-file bench_$tag.json 2>&1 | grep -E "Successful|Benchmark duration" | head -2
}

echo "=== RUN 1: TRITON_ATTN baseline ==="
pkill -f api_server 2>/dev/null; sleep 6
launch baseline base || exit 1
warm
echo "--- bench short (512/256) ---"; bench 512 256 base_short
echo "--- bench long-ctx (16384/256) ---"; bench 16384 256 base_long
echo "--- bench second short pass (stability check) ---"; bench 512 256 base_short2

echo "=== RUN 2: ROCM_AITER_UNIFIED_ATTN ==="
pkill -f api_server 2>/dev/null; sleep 6
launch aiter ua || exit 1
warm
echo "--- bench short (512/256) ---"; bench 512 256 ua_short
echo "--- bench long-ctx (16384/256) ---"; bench 16384 256 ua_long
echo "--- bench second short pass ---"; bench 512 256 ua_short2

echo "=== ALL DONE ==="
touch /tmp/e2e_v2_done
