#!/bin/bash
# Multi-map model with single NMDA_avg biological map
# Tests the multi-map framework with one additional biological map

filename="multi_map_optim.py"
model_name="multimap"
n_particles=300
n_tasks=1  
output_directory="baseline_myelin_nmda"

# Threading settings (MacBook Pro M3 14 Cores (10P; 4E))
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3


echo "=== Running Multi-Map Model (NMDA_avg & Myelin) ==="
echo "Expected parameters: w_EI_bias, w_EI_coeff1, w_EI_coeff2, w_EE_bias, w_EE_coeff1, w_EE_coeff2, G (7 total)"
echo "Maps: T1w/T2w (inverted), NMDA_avg (direct) - both linearized"
echo "Output: outputs/${output_directory}/"

for iter in {1..10}
do
for samplers in {0..4}
do
python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory t1wt2w,NMDA_avg linearize &
done
wait
python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory t1wt2w,NMDA_avg linearize
done

echo "=== Multi-map NMDA model optimization completed ==="