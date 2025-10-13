#!/bin/bash
# Surrogate Maps Optimization Script
# Processes 100 surrogate maps for a selected category
# OPTIMIZED FOR: 96-core Intel Xeon Gold 5418Y cluster

echo "=========================================="
echo "Starting Surrogate Maps Optimization"
echo "=========================================="

# ===== EDIT THIS SECTION =====
# Select which category of surrogate maps to process
CATEGORY="myelin"  # Options: myelin, gaba, nmda, dopamine, norepinephrine

# Optimization parameters
GLOBAL_ITERATIONS=50  # Reduced for surrogate maps (null models)
n_particles=25        # 25 particles × 4 samplers = 100 total
n_samplers=4         # Using 4 samplers instead of 5
# =============================

# Record start time
start_time=$(date)
echo "Started at: $start_time"

# Script settings
filename="multi_map_optim.py"
model_name="multimap"
n_tasks=1

# CLUSTER THREADING SETTINGS
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Paths (now that surrogate_maps is in data/maps/)
SURROGATE_DIR="../data/maps/surrogate_maps/${CATEGORY}"

echo "CONFIGURATION:"
echo "  Category: $CATEGORY"
echo "  Surrogate directory: $SURROGATE_DIR"
echo "  Iterations per map: $GLOBAL_ITERATIONS"
echo "  Particles per sampler: $n_particles"
echo "  Number of samplers: $n_samplers"
echo "  Total particles: $((n_particles * n_samplers))"
echo "  Threading per sampler: $OPENBLAS_NUM_THREADS threads"
echo ""

# Validate surrogate directory exists
if [[ ! -d "$SURROGATE_DIR" ]]; then
    echo "ERROR: Surrogate directory not found: $SURROGATE_DIR"
    echo "Available categories:"
    ls -1 "../data/maps/surrogate_maps/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Count available surrogate files
surrogate_count=$(ls -1 "${SURROGATE_DIR}/${CATEGORY}_surrogate_"*.pscalar.nii 2>/dev/null | wc -l)
echo "Found $surrogate_count surrogate files in $CATEGORY category"

if [[ $surrogate_count -eq 0 ]]; then
    echo "ERROR: No surrogate files found matching pattern: ${CATEGORY}_surrogate_*.pscalar.nii"
    echo "Available categories:"
    ls -1 "../data/maps/surrogate_maps/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo ""

# Function to run one iteration for a specific surrogate map
run_surrogate_iteration() {
    local surrogate_filename=$1  # Just the filename without extension
    local iteration=$2
    local surrogate_num=$3
    local output_dir="${CATEGORY}/surrogate_$(printf '%03d' $surrogate_num)"
    
    # The path to pass to Python script (relative to data/maps/)
    local surrogate_path="surrogate_maps/${CATEGORY}/${surrogate_filename}"
    
    echo "    Running iteration $iteration for surrogate $surrogate_num"
    
    # Run parallel samplers
    for ((s=0; s<n_samplers; s++))
    do
        python $filename $model_name $n_particles $n_tasks $s sampler $output_dir $surrogate_path linearize &
    done
    
    # Wait for all samplers to complete
    wait
    
    # Run wrapper
    if ! python $filename $model_name $n_particles $n_tasks 0 wrapper $output_dir $surrogate_path linearize; then
        echo "⚠️  Wrapper failed for iteration $iteration, surrogate $surrogate_num"
        echo "    This may indicate particle degeneracy - consider this surrogate map problematic"
        return 1  # Return error code instead of break
    fi
    return 0  # Success
}

# Add this function before the main loop
check_surrogate_completion() {
    local surrogate_num=$1
    local output_dir="../outputs/${CATEGORY}/surrogate_$(printf '%03d' $surrogate_num)"
    local final_iteration_file="${output_dir}/iteration_${GLOBAL_ITERATIONS}.hdf5"
    
    if [[ -f "$final_iteration_file" ]]; then
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# Main processing loop
echo "Starting surrogate map processing..."
echo "=========================================="

for surrogate_num in {1..100}; do
    surrogate_filename="${CATEGORY}_surrogate_$(printf '%03d' $surrogate_num)"
    surrogate_file_path="${SURROGATE_DIR}/${surrogate_filename}.pscalar.nii"
    
    # Check if this surrogate file exists
    if [[ ! -f "$surrogate_file_path" ]]; then
        echo "⚠️  Skipping surrogate $surrogate_num - file not found"
        continue
    fi
    
    # Check if already completed
    if check_surrogate_completion "$surrogate_num"; then
        echo "✅ Skipping surrogate $surrogate_num - already completed ($GLOBAL_ITERATIONS iterations)"
        continue
    fi
    
    echo "--- Processing Surrogate $surrogate_num/100 ($CATEGORY) ---"
    echo "File: ${surrogate_filename}.pscalar.nii"
    echo "Output: outputs/${CATEGORY}/surrogate_$(printf '%03d' $surrogate_num)/"
    echo "Progress: $surrogate_num/100 ($(( (surrogate_num * 100) / 100 ))%)"
    
    # Run all iterations for this surrogate map
    for ((iteration=1; iteration<=GLOBAL_ITERATIONS; iteration++)); do
        echo "  Iteration $iteration/$GLOBAL_ITERATIONS"
        if ! run_surrogate_iteration "$surrogate_filename" "$iteration" "$surrogate_num"; then
            echo "❌ Surrogate $surrogate_num failed at iteration $iteration - skipping to next surrogate"
            break  # Now this break is in the right context
        fi
    done
    
    echo "✓ Surrogate $surrogate_num completed ($GLOBAL_ITERATIONS iterations)"
    echo ""
done

# Summary
end_time=$(date)
echo "=========================================="
echo "Surrogate maps optimization completed!"
echo "Category: $CATEGORY"
echo "Started: $start_time"
echo "Ended: $end_time"
echo ""
echo "Results stored in: outputs/${CATEGORY}/"
echo "  - surrogate_001/ through surrogate_100/"
echo ""
echo "Summary:"
echo "  Category: $CATEGORY"
echo "  Surrogate maps processed: up to 100"
echo "  Iterations per map: $GLOBAL_ITERATIONS"
echo "  Particles per sampler: $n_particles"
echo "  Total particles: $((n_particles * n_samplers)) ($n_samplers samplers)"
echo "  Parameters optimized: 5 (w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G)"
echo "==========================================" 