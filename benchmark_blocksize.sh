#!/bin/bash
set -eo pipefail
# Benchmark block size performance across different array sizes

echo "Running block size experiments..."
echo ""

for size in 10000 100000 1000000 10000000 100000000; do
    size_name=""
    case $size in
        10000) size_name="10K" ;;
        100000) size_name="100K" ;;
        1000000) size_name="1M" ;;
        10000000) size_name="10M" ;;
        100000000) size_name="100M" ;;
    esac

    for block in 1 2 4 8 16 32 64 128 256 512 1024; do
        echo -n "Size: $size_name, Block: $block ... "
        median=$(./vector -n $size -b $block 2>&1 | grep "Median:" | awk '{print $2}')
        echo "$median"
    done
done
