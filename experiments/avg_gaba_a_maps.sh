data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

wb_command -cifti-average \
  ${data_dir}/gaba_a_avg.pscalar.nii \
  -cifti \
  ${data_dir}/gaba_a/GABRA1.pscalar.nii \
  -cifti \
  ${data_dir}/gaba_a/GABRA2.pscalar.nii \
  -cifti \
  ${data_dir}/gaba_a/GABRA3.pscalar.nii \
  -cifti \
  ${data_dir}/gaba_a/GABRA4.pscalar.nii \
  -cifti \
  ${data_dir}/gaba_a/GABRA5.pscalar.nii 