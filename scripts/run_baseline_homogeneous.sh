#!/bin/bash
# Baseline homogeneous model optimization
# No spatial heterogeneity - all regions have identical parameters

filename="multi_map_optim.py"
model_name="homogeneous"
n_particles=300
n_tasks=1
n_iterations=8
output_directory="baseline_homogeneous"

# Threading settings (MacBook Pro M3 14 Cores (10P; 4E))
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

echo "=== Running Baseline Homogeneous Model ==="
echo "Expected parameters: w_EI, w_EE, G (3 total)"
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

echo "=== Homogeneous model optimization completed ==="