#!/bin/bash
# Multi-receptor optimization script - runs all receptor maps in parallel rounds
# Each receptor progresses at the same rate (one iteration per round)

echo "=========================================="
echo "Starting Multi-Receptor Optimization"
echo "=========================================="

# Record start time
start_time=$(date)
echo "Started at: $start_time"

# Global settings
filename="multi_map_optim.py"
model_name="multimap"
n_particles=300
n_tasks=1
GLOBAL_ITERATIONS=2  # Set this to control total iterations for all receptors

# Threading settings (MacBook Pro M3 14 Cores (10P; 4E))
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

# Define receptor maps (without .pscalar.nii extension)
declare -a receptors=(
    # "dopamine_avg"
    # "serotonin_avg"
    "norepinephrine_avg"
    # "nicotinic_avg"
    # "muscarinic_avg"
    # "gaba_a_avg"
    "nmda_avg"
)

# Function to run one iteration for a specific receptor
run_receptor_iteration() {
    local receptor=$1
    local iteration=$2
    local output_dir="receptor_${receptor}"
    
    echo "  Running $receptor (iteration $iteration)"
    
    # Run parallel samplers
    for samplers in {0..4}
    do
        python $filename $model_name $n_particles $n_tasks $samplers sampler $output_dir $receptor linearize &
    done
    wait
    
    # Run wrapper
    python $filename $model_name $n_particles $n_tasks 0 wrapper $output_dir $receptor linearize
}

# Print configuration
echo "Global iterations: $GLOBAL_ITERATIONS"
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
        echo "--- $receptor (Round $round) ---"
        echo "Expected parameters: w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G (5 total)"
        echo "Map: $receptor (linearized)"
        echo "Output: outputs/receptor_${receptor}/"
        
        run_receptor_iteration "$receptor" "$round"
        
        echo "✓ $receptor round $round completed"
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
    echo "  - outputs/receptor_${receptor}/"
done
echo ""
echo "Receptor Summary:"
echo "  Total receptors: ${#receptors[@]}"
echo "  Iterations per receptor: $GLOBAL_ITERATIONS"
echo "  Parameters per receptor: 5 (w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G)"
echo "  Maps used: linearized averaged receptor maps"
echo "==========================================" 