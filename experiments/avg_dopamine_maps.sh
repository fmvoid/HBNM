data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

wb_command -cifti-average \
  ${data_dir}/dopamine_avg.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD1.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD2.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD3.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD4.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD5.pscalar.nii 