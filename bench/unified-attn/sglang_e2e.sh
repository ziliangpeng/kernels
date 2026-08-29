#!/bin/bash
# SGLang e2e A/B: aiter unified attention ON vs OFF (legacy ragged path)
set -u
MODEL=Qwen/Qwen3-1.7B
PORT=18999

launch() {
  local mode=$1 tag=$2
  envs=""
  if [ "$mode" = "unified" ]; then
    envs="SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1"
  else
    envs="SGLANG_USE_AITER=1"
  fi
  echo "[$(date +%T)] launching $mode..."
  env $envs nohup python3 -m sglang.launch_server \
    --model-path $MODEL --port $PORT --dtype bfloat16 \
    --context-length 8192 --mem-fraction-static 0.8 \
    --attention-backend aiter \
    --cuda-graph-max-bs 64 \
    > /tmp/sgl_serve_$tag.log 2>&1 &
  for i in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || true)
    if [ "$code" = "200" ]; then echo "[$(date +%T)] $mode ready (waited ${i}x5s)"; return 0; fi
    sleep 5
  done
  echo "TIMEOUT waiting for $mode"; return 1
}

warm() {
  echo "  warming..."
  for lens in "128 128" "512 256" "1024 256" "2048 128" "4096 256" "8192 256"; do
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
}

bench() {
  local in=$1 out=$2 tag=$3
  python3 -m sglang.bench_serving \
    --backend sglang --model $MODEL --base-url http://localhost:$PORT \
    --dataset-name random-input-len --random-input-len $in --random-output-len $out \
    --num-prompts 96 --max-concurrency 32 --seed 42 \
    --result-dir /tmp --result-file sgl_$tag.json 2>&1 | grep -E "Successful|Duration|Output token throughput|Median TTFT|Median ITL|Mean TPOT" | head -8
}

echo "=== RUN 1: aiter unified OFF (legacy ragged) ==="
launch legacy leg || exit 1
warm
echo "--- short (512/256) ---"; bench 512 256 leg_short
echo "--- long (16384/256) ---"; bench 16384 256 leg_long

echo "=== RUN 2: aiter unified ON ==="
pkill -f sglang.launch_server 2>/dev/null; sleep 8
launch unified uni || exit 1
warm
echo "--- short (512/256) ---"; bench 512 256 uni_short
echo "--- long (16384/256) ---"; bench 16384 256 uni_long

echo "=== ALL DONE ==="
touch /tmp/sgl_e2e_done
