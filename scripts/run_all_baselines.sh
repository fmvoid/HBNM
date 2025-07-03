#!/bin/bash
# Master script to run all baseline comparisons overnight

echo "=========================================="
echo "Starting Complete Baseline Comparison"
echo "=========================================="

# Record start time
start_time=$(date)
echo "Started at: $start_time"

# Run homogeneous baseline
# echo -e "\n[1/4] Running Homogeneous Model..."
# ./run_baseline_homogeneous.sh

# Run heterogeneous baseline  
echo -e "\n[2/4] Running Heterogeneous Model..."
./run_baseline_heterogeneous.sh

# Run single NMDA map baseline
echo -e "\n[3/4] Running Single NMDA Map Model..."
./run_baseline_nmda.sh

# Run dual map baseline (T1w/T2w + NMDA)
echo -e "\n[4/4] Running Dual Map Model..."
./run_baseline_multimap.sh

# Summary
end_time=$(date)
echo "=========================================="
echo "All baseline optimizations completed!"
echo "Started: $start_time"
echo "Ended: $end_time"
echo ""
echo "Results stored in:"
echo "  - outputs/baseline_homogeneous/"
echo "  - outputs/baseline_heterogeneous/"  
echo "  - outputs/baseline_nmda/"
echo "  - outputs/baseline_myelin_nmda/"
echo ""
echo "Model Comparison Summary:"
echo "  1. Homogeneous:     3 params, no spatial heterogeneity"
echo "  2. Heterogeneous:   5 params, T1w/T2w map only"
echo "  3. Single NMDA:     5 params, NMDA_avg map only"
echo "  4. Dual Map:        7 params, T1w/T2w + NMDA_avg maps"
echo "=========================================="