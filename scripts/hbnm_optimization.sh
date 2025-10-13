#!/bin/bash
# OPTIMIZED Multi-receptor optimization script for Muncher
# Runs receptor maps in parallel rounds
# OPTIMIZED FOR: 96-core Intel Xeon Gold 5418Y cluster with 251GB RAM

echo "=========================================="
echo "Starting OPTIMIZED SINGLE-Receptor Optimization"
echo "=========================================="

# Record start time
start_time=$(date)
echo "Started at: $start_time"

# CLUSTER-OPTIMIZED SETTINGS
filename="multi_map_optim.py"
model_name="multimap"
n_particles=100
n_tasks=1
GLOBAL_ITERATIONS=20

# OPTIMIZED THREADING SETTINGS for 96-core cluster
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "PERFORMANCE OPTIMIZATION SETTINGS:"
echo "  - CPU cores available: $(nproc)"
echo "  - OpenBLAS threads: $OPENBLAS_NUM_THREADS"
echo "  - Particles per sampler: $n_particles"
echo "  - Total particles: $((n_particles * 5)) (5 samplers)"
echo ""

# Define receptor maps (relative to data/maps/ directory)
declare -a receptors=(
    "dopamine_d1like_avg"
    "dopamine_d2like_avg"
    )

# Validation function for receptor maps
validate_receptor_map() {
    local pscalar_file="../data/maps/${1}.pscalar.nii"
    
    if [[ ! -f "$pscalar_file" ]]; then
        echo "ERROR: File not found: ${1}.pscalar.nii - check file format"
        exit 1
    fi
}

# Function to run one iteration for a specific receptor
run_receptor_iteration() {
    local receptor=$1
    local iteration=$2
    local output_dir="${receptor##*/}"  # Use basename for output directory
    
    echo "  Running $receptor (iteration $iteration)"
    
    # Run parallel samplers (5 samplers × 16 threads each = 80 threads total)
    for samplers in {0..4}
    do
        python $filename $model_name $n_particles $n_tasks $samplers sampler $output_dir $receptor linearize &
    done
    
    # Wait for all samplers to complete
    wait
    
    # Run wrapper
    python $filename $model_name $n_particles $n_tasks 0 wrapper $output_dir $receptor linearize
}

# Validate all receptor maps before starting
echo "Validating receptor maps..."
for receptor in "${receptors[@]}"; do
    validate_receptor_map "$receptor"
done
echo "✓ All receptor maps validated"
echo ""

# Print configuration
echo "OPTIMIZED CONFIGURATION:"
echo "  Global iterations: $GLOBAL_ITERATIONS"
echo "  Particles per sampler: $n_particles"
echo "  Total particles: $((n_particles * 5)) (5 samplers)"
echo "  Threading: $OPENBLAS_NUM_THREADS threads per math library"
echo "  Total compute capacity: 5 samplers × $OPENBLAS_NUM_THREADS threads = $((5 * OPENBLAS_NUM_THREADS)) threads"
echo ""
echo "Receptor maps to optimize:"
for receptor in "${receptors[@]}"; do
    echo "  - $receptor"
done
echo ""

# Main optimization loop - all receptors progress at same rate
for ((round=1; round<=GLOBAL_ITERATIONS; round++))
do
    echo "=========================================="
    echo "Round $round/$GLOBAL_ITERATIONS"
    echo "=========================================="
    
    # Run one iteration for each receptor in this round
    for receptor in "${receptors[@]}"; do
        receptor_name=${receptor##*/}  # Extract filename without path
        echo "--- $receptor_name (Round $round) ---"
        echo "Expected parameters: w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G (5 total)"
        echo "Map: data/maps/$receptor.pscalar.nii (linearized)"
        echo "Output: outputs/${receptor_name}/"
        echo "Particles: $n_particles per sampler × 5 samplers = $((n_particles * 5)) total"
        
        # Determine expected inversion based on receptor type
        if [[ "$receptor_name" == *"gaba"* ]]; then
            echo "Expected inversion: TRUE (GABA should be inverted)"
        else
            echo "Expected inversion: FALSE (dopamine/nmda/norepinephrine should be direct)"
        fi
        
        run_receptor_iteration "$receptor" "$round"
        
        echo "✓ $receptor_name round $round completed"
        echo ""
    done
    
    echo "Round $round completed for all receptors"
    echo ""
done

# Summary
end_time=$(date)
echo "=========================================="
echo "All receptor optimizations completed!"
echo "Started: $start_time"
echo "Ended: $end_time"
echo ""
echo "Results stored in:"
for receptor in "${receptors[@]}"; do
    receptor_name=${receptor##*/}
    echo "  - outputs/${receptor_name}/"
done
echo ""
echo "OPTIMIZED Receptor Summary:"
echo "  Total receptors: ${#receptors[@]}"
echo "  Iterations per receptor: $GLOBAL_ITERATIONS"
echo "  Parameters per receptor: 5 (w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G)"
echo "  Particles per sampler: $n_particles"
echo "  Total particles: $((n_particles * 5)) (5 samplers)"
echo "  Threading per sampler: $OPENBLAS_NUM_THREADS threads"
echo "  Maps used: linearized receptor maps from data/maps/"
echo "==========================================" 