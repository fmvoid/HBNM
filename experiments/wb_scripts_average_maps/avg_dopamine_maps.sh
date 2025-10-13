data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

# Create overall dopamine average (all receptors)
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

# Create D1-like map (DRD1 & DRD5) - Predominantly excitatory, direct pathway
wb_command -cifti-average \
  ${data_dir}/dopamine_d1like_avg.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD1.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD5.pscalar.nii

# Create D2-like map (DRD2/3/4) - Mostly inhibitory, indirect pathway
wb_command -cifti-average \
  ${data_dir}/dopamine_d2like_avg.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD2.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD3.pscalar.nii \
  -cifti \
  ${data_dir}/dopamine/DRD4.pscalar.nii 