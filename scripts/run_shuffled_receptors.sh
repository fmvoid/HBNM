#!/bin/bash
# Multi-receptor optimization script - runs all SHUFFLED receptor maps in parallel rounds
# Each receptor progresses at the same rate (one iteration per round)
# This script tests the specificity of receptor effects by using shuffled control maps

echo "=========================================="
echo "Starting Multi-Receptor Optimization (SHUFFLED CONTROLS)"
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

# Define shuffled receptor maps (with shuffled/ subdirectory prefix)
declare -a shuffled_receptors=(
    "shuffled/dopamine_shuffled"
    "shuffled/gaba_shuffled"
    "shuffled/nmda_shuffled"
    "shuffled/norepinephrine_shuffled"
)

# Function to run one iteration for a specific shuffled receptor
run_shuffled_receptor_iteration() {
    local receptor=$1
    local iteration=$2
    local output_dir="shuffled_${receptor##*/}"  # Remove 'shuffled/' prefix for output directory
    
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
echo "Shuffled receptor maps to optimize:"
for receptor in "${shuffled_receptors[@]}"; do
    echo "  - $receptor"
done
echo ""
echo "Note: Using shuffled control maps to test specificity of receptor effects"
echo "These results should show reduced model performance compared to real maps"
echo ""

# Main optimization loop - all shuffled receptors progress at same rate
for ((round=1; round<=GLOBAL_ITERATIONS; round++))
do
    echo "=========================================="
    echo "Round $round/$GLOBAL_ITERATIONS (SHUFFLED CONTROLS)"
    echo "=========================================="
    
    # Run one iteration for each shuffled receptor in this round
    for receptor in "${shuffled_receptors[@]}"; do
        receptor_name=${receptor##*/}  # Extract filename without path
        echo "--- $receptor_name (Round $round) ---"
        echo "Expected parameters: w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G (5 total)"
        echo "Map: $receptor (shuffled control, linearized)"
        echo "Output: outputs/shuffled_${receptor_name}/"
        
        # Determine expected inversion based on receptor type
        if [[ "$receptor_name" == *"gaba"* ]]; then
            echo "Expected inversion: TRUE (GABA should be inverted)"
        else
            echo "Expected inversion: FALSE (dopamine/nmda/norepinephrine should be direct)"
        fi
        
        run_shuffled_receptor_iteration "$receptor" "$round"
        
        echo "✓ $receptor_name round $round completed"
        echo ""
    done
    
    echo "Round $round completed for all shuffled receptors"
    echo ""
done

# Summary
end_time=$(date)
echo "=========================================="
echo "All shuffled receptor optimizations completed!"
echo "Started: $start_time"
echo "Ended: $end_time"
echo ""
echo "Results stored in:"
for receptor in "${shuffled_receptors[@]}"; do
    receptor_name=${receptor##*/}
    echo "  - outputs/shuffled_${receptor_name}/"
done
echo ""
echo "Shuffled Receptor Summary:"
echo "  Total shuffled receptors: ${#shuffled_receptors[@]}"
echo "  Iterations per receptor: $GLOBAL_ITERATIONS"
echo "  Parameters per receptor: 5 (w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G)"
echo "  Maps used: linearized shuffled receptor control maps"
echo ""
echo "IMPORTANT: These are SHUFFLED CONTROL maps"
echo "- They should show reduced model performance vs real receptor maps"
echo "- Use these results to assess specificity of receptor effects"
echo "- Compare with outputs from run_all_receptors.sh for validation"
echo "==========================================" 