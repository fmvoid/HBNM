filename="model_optimization.py"
model_name="heterogeneous"
n_particles=25
n_tasks=1
n_iterations=2
output_directory="heterogeneous"

for iter in {1..1}
do
for samplers in {0..1}
do
python $filename $model_name $n_particles $n_tasks $samplers sampler $output_directory &
done
wait
python $filename $model_name $n_particles $n_tasks 0 wrapper $output_directory
done

