#!/bin/bash
# Baseline heterogeneous model optimization  
# Single T1w/T2w map for spatial heterogeneity

filename="multi_map_optim.py"
model_name="heterogeneous" 
n_particles=300
n_tasks=1
n_iterations=8
output_directory="baseline_heterogeneous"

# Threading settings (MacBook Pro M3 14 Cores (10P; 4E))
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

echo "=== Running Baseline Heterogeneous Model (T1w/T2w) ==="
echo "Expected parameters: w_EI_bias, w_EI_slope, w_EE_bias, w_EE_slope, G (5 total)"
echo "Output: outputs/${output_directory}/"

for iter in {1..10}
do
for samplers in {0..4}
do
python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory &
done
wait
python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory
done

echo "=== Heterogeneous model optimization completed ==="