#!/bin/bash
# Single T1w/T2w map for spatial heterogeneity

# CLUSTER THREADING SETTINGS
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

filename="optimization_tms_fmri_data.py"
model_name="multimap" 
n_particles=100
n_tasks=1
output_directory="TEHC_Myelin_Group"
fc_vector_path="/home/frank/TMS_fMRI/fc_output/group_means/TEHC_mean_fc.npy"
maps_path="/home/frank/HBNM/data/heterogeneity_vectors/linearized/myelin_linearized.npy"

echo "=== Running Heterogeneous Model (T1w/T2w) ==="
echo "Expected parameters: w_EI_bias, w_EI_slope, w_EE_bias, w_EE_slope, G (5 total)"
echo "Output: outputs/${output_directory}/"

for iter in {0..47}
do
for samplers in {0..4}
do
python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory $fc_vector_path $maps_path no_linearize invert &
done
wait
python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory $fc_vector_path $maps_path no_linearize invert
done

echo "=== Heterogeneous model optimization completed ==="