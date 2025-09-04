#!/bin/bash
# Baseline heterogeneous model optimization  
# Single T1w/T2w map for spatial heterogeneity

filename="multi_map_optim.py"
model_name="multimap" 
n_particles=100
n_tasks=1
output_directory="Heterogeneous_Dem_Range"

# CLUSTER THREADING SETTINGS
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6


echo "=== Running Baseline Heterogeneous Model (T1w/T2w) ==="
echo "Expected parameters: w_EI_bias, w_EI_slope, w_EE_bias, w_EE_slope, G (5 total)"
echo "Output: outputs/${output_directory}/"

for iter in {0..48}
do
for samplers in {0..5}
do
python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory t1wt2w &
done
wait
python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory t1wt2w
done

echo "=== Heterogeneous model optimization completed ==="