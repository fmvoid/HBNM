#!/bin/bash
# Master script to run all baseline comparisons overnight

echo "=========================================="
echo "Starting Complete Baseline Comparison"
echo "Total estimated time: ~75-90 minutes"
echo "=========================================="

# Record start time
start_time=$(date)
echo "Started at: $start_time"

# Run homogeneous baseline
echo -e "\n[1/3] Running Homogeneous Model..."
./run_baseline_homogeneous.sh

# Run heterogeneous baseline  
echo -e "\n[2/3] Running Heterogeneous Model..."
./run_baseline_heterogeneous.sh

# Run multi-map NMDA
echo -e "\n[3/3] Running Multi-map NMDA Model..."
./run_multimap_nmda.sh

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
echo "  - outputs/multimap_nmda_only/"
echo "=========================================="