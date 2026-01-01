#!/bin/bash
set -eo pipefail
# Test VMA fused vs separate sequentially for accuracy

echo "Running VMA tests sequentially for accuracy..."
echo ""

for size in 1000 10000 100000 1000000 10000000 100000000; do
    size_name=""
    case $size in
        1000) size_name="1K" ;;
        10000) size_name="10K" ;;
        100000) size_name="100K" ;;
        1000000) size_name="1M" ;;
        10000000) size_name="10M" ;;
        100000000) size_name="100M" ;;
    esac

    echo "=== $size_name elements ==="
    sep=$(./vector -n $size --mode vma 2>&1 | grep "Median:" | awk '{print $2}')
    echo "  Separate: $sep"
    fus=$(./vector -n $size --mode vma --fused 2>&1 | grep "Median:" | awk '{print $2}')
    echo "  Fused:    $fus"
    echo ""
done
