#!/bin/bash

echo "=========================================="
echo "CUDA vs TinyGrad Comprehensive Benchmark"
echo "Environment: H100 80GB, CUDA 12.4, sm_90"
echo "=========================================="
echo ""

# Array sizes to test (100K, 1M, 10M, 100M, 1B)
sizes=(100000 1000000 10000000 100000000 1000000000)

echo "Operation 1: VMA (d = a * b + c)"
echo "------------------------------------------------------------------------"
printf "%-10s | %-15s | %-15s | %s\n" "Size" "CUDA Fused" "TinyGrad" "Slowdown"
echo "-----------|-----------------|-----------------|------------"

for size in "${sizes[@]}"; do
    # CUDA fused
    cuda_result=$(./vector -n $size --mode vma --fused 2>&1 | grep "Median:" | awk '{print $2}')

    # TinyGrad
    tinygrad_result=$(uv run tinygrad_comparison.py -n $size -o vma 2>&1 | grep "Median:" | awk '{print $2}')

    # Calculate slowdown
    slowdown=$(echo "scale=1; $tinygrad_result / $cuda_result" | bc)

    # Format size
    if [ $size -ge 1000000 ]; then
        size_str="$(($size / 1000000))M"
    elif [ $size -ge 1000 ]; then
        size_str="$(($size / 1000))K"
    else
        size_str="$size"
    fi

    printf "%-10s | %-15s | %-15s | %.1fx\n" "$size_str" "$cuda_result" "$tinygrad_result" "$slowdown"
done

echo ""
echo "Operation 2: Reduction (sum)"
echo "------------------------------------------------------------------------"
printf "%-10s | %-15s | %-15s | %s\n" "Size" "CUDA Warp-opt" "TinyGrad" "Slowdown"
echo "-----------|-----------------|-----------------|------------"

for size in "${sizes[@]}"; do
    # CUDA warp-optimized threshold
    cuda_result=$(./reduce -n $size --method threshold --warp-opt 2>&1 | grep "Median:" | awk '{print $2}')

    # TinyGrad
    tinygrad_result=$(uv run tinygrad_comparison.py -n $size -o reduce 2>&1 | grep "Median:" | awk '{print $2}')

    # Calculate slowdown
    slowdown=$(echo "scale=1; $tinygrad_result / $cuda_result" | bc)

    # Format size
    if [ $size -ge 1000000 ]; then
        size_str="$(($size / 1000000))M"
    elif [ $size -ge 1000 ]; then
        size_str="$(($size / 1000))K"
    else
        size_str="$size"
    fi

    printf "%-10s | %-15s | %-15s | %.1fx\n" "$size_str" "$cuda_result" "$tinygrad_result" "$slowdown"
done

echo ""
echo "Operation 3: Softmax"
echo "------------------------------------------------------------------------"
printf "%-10s | %-15s | %-15s | %s\n" "Size" "CUDA Multi-pass" "TinyGrad" "Slowdown"
echo "-----------|-----------------|-----------------|------------"

for size in "${sizes[@]}"; do
    # CUDA multi-pass
    cuda_result=$(./softmax -n $size --method multi 2>&1 | grep "Median:" | awk '{print $2}')

    # TinyGrad (extract built-in softmax result)
    tinygrad_result=$(uv run tinygrad_comparison.py -n $size -o softmax 2>&1 | grep -A 10 "Built-in softmax" | grep "Median:" | awk '{print $2}')

    # Calculate slowdown
    slowdown=$(echo "scale=1; $tinygrad_result / $cuda_result" | bc)

    # Format size
    if [ $size -ge 1000000 ]; then
        size_str="$(($size / 1000000))M"
    elif [ $size -ge 1000 ]; then
        size_str="$(($size / 1000))K"
    else
        size_str="$size"
    fi

    printf "%-10s | %-15s | %-15s | %.1fx\n" "$size_str" "$cuda_result" "$tinygrad_result" "$slowdown"
done

echo ""
echo "=========================================="
echo "Summary:"
echo "  - TinyGrad adds significant Python/framework overhead"
echo "  - Hand-optimized CUDA is much faster for these operations"
echo "  - Trade-off: Productivity vs Performance"
echo "=========================================="
