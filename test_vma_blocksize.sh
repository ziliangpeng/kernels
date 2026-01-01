#!/bin/bash
# Test VMA fused kernel with different block sizes

echo "Testing VMA fused with different block sizes (10M elements)..."
echo ""

for bs in 1 2 4 8 16 32 64 128 256 512 1024; do
    echo -n "Block size $bs: "
    ./vector -n 10000000 --mode vma --fused -b $bs 2>&1 | grep "Median:" | awk '{print $2}'
done
