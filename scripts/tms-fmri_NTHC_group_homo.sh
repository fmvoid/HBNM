#!/bin/bash
# Homogeneous model optimization

# CLUSTER THREADING SETTINGS
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

filename="optimization_tms_fmri_data.py"
model_name="multimap" 
n_particles=100
n_tasks=1
output_directory="NTHC_Homo_Group"
fc_vector_path="/home/frank/TMS_fMRI/fc_output/group_means/NTHC_mean_fc.npy"
maps_path="None"

echo "=== Running Homogeneous Model ==="
echo "Output: outputs/${output_directory}/"

start_time=$(date +%s)

for iter in {0..30}
do
for samplers in {0..4}
do
  (time python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory $fc_vector_path $maps_path no_linearize invert "0.001,5.0;0.001,15.0;0.001,5.0") &
done
wait
time python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory $fc_vector_path $maps_path no_linearize invert "0.001,5.0;0.001,15.0;0.001,5.0"
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "=== Homogeneous model optimization completed ==="
echo "Total elapsed time: ${elapsed} seconds"