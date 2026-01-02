#!/bin/bash
# Comprehensive reduction benchmark: multiple array sizes × multiple methods/thresholds

echo "Reduction Performance Benchmark"
echo "Testing GPU vs Threshold methods with various configurations"
echo ""

# Array sizes: 1K to 1B
sizes=(1000 10000 100000 1000000 10000000 100000000 1000000000)
size_names=("1K" "10K" "100K" "1M" "10M" "100M" "1B")

# Threshold values to test
thresholds=(1 10 100 1000 10000 100000)

# Print header
printf "%-10s | %-12s |" "Array Size" "GPU (full)"
for thresh in "${thresholds[@]}"; do
    printf " %-15s |" "Threshold=$thresh"
done
printf "\n"

# Print separator
printf "%-10s-+-%-12s-+" "----------" "------------"
for thresh in "${thresholds[@]}"; do
    printf "-%-15s-+" "---------------"
done
printf "\n"

# Run benchmarks
for i in "${!sizes[@]}"; do
    size=${sizes[$i]}
    size_name=${size_names[$i]}

    printf "%-10s |" "$size_name"

    # Test GPU method (fully recursive)
    gpu_time=$(./reduce -n $size --method gpu 2>&1 | grep "Median:" | awk '{print $2}')
    printf " %-12s |" "$gpu_time ms"

    # Test threshold method with different thresholds
    for thresh in "${thresholds[@]}"; do
        thresh_time=$(./reduce -n $size --method threshold --cpu-threshold $thresh 2>&1 | grep "Median:" | awk '{print $2}')
        printf " %-15s |" "$thresh_time ms"
    done

    printf "\n"
done

echo ""
echo "Benchmark complete!"
echo ""
echo "Key:"
echo "  GPU (full)     = Fully recursive GPU (reduce until 1 element)"
echo "  Threshold=N    = GPU reduce until <= N elements, then CPU"
