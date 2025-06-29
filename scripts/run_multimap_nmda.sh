#!/bin/bash
# Multi-map model with single NMDA_avg biological map
# Tests the multi-map framework with one additional biological map

filename="multi_map_optim.py"
model_name="multimap"
n_particles=100
n_tasks=1  
n_iterations=8
output_directory="multimap_nmda_only"

echo "=== Running Multi-Map Model (NMDA_avg only) ==="
echo "Expected parameters: w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G (5 total)"
echo "Maps: NMDA_avg (linearized)"
echo "Output: outputs/${output_directory}/"

for iter in {1..$n_iterations}
do
for samplers in {0..1}
do
python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory NMDA_avg linearize &
done
wait
python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory NMDA_avg linearize
done

echo "=== Multi-map NMDA model optimization completed ==="