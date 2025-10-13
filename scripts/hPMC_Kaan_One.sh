#!/bin/bash
# Hetereogeneous model optimization

# CLUSTER THREADING SETTINGS
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

filename="optimization_tms_fmri_data.py"
model_name="multimap" 
n_particles=100
n_tasks=1
output_directory="Kaan_One"
# This is a folder that contains .npy files
fc_vector_path="/home/frank/HBNM/data/simulated_wc_data/arr1_timeseries"
maps_path="/home/frank/HBNM/data/heterogeneity_vectors/linearized/myelin_linearized.npy"

# Custom priors for heterogeneous model: w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G
custom_priors="0.001,10.0;0.0,10.0;0.001,10.0;0.0,10.0;0.01,5.0" # M6
# custom_priors="0.001,10.0;0.0,10.0;0.001,10.0;0.0,10.0;2.9,3.1" # M5
# custom_priors="0.001,2.0;0.0,2.5;0.001,5.0;0.0,15.0;0.001"

echo "Output: outputs/${output_directory}/"
echo "Priors: ${custom_priors}"

start_time=$(date +%s)

for iter in {0..35}
do
for samplers in {0..4}
do
  (time python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory $fc_vector_path $maps_path no_linearize invert "${custom_priors}") &
done
wait
time python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory $fc_vector_path $maps_path no_linearize invert "${custom_priors}"
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "=== Hetereogeneous model optimization completed ==="
echo "Total elapsed time: ${elapsed} seconds"