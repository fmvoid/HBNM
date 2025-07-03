data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

wb_command -cifti-average \
  ${data_dir}/serotonin_avg.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR1A.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR1E.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR1F.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR2A.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR2C.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR3B.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR4.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR5A.pscalar.nii \
  -cifti \
  ${data_dir}/serotonin/HTR7.pscalar.nii 