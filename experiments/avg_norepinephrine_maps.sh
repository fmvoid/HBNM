data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

wb_command -cifti-average \
  ${data_dir}/norepinephrine_avg.pscalar.nii \
  -cifti \
  ${data_dir}/norepinephrine/ADRA1A.pscalar.nii \
  -cifti \
  ${data_dir}/norepinephrine/ADRA1B.pscalar.nii \
  -cifti \
  ${data_dir}/norepinephrine/ADRA1D.pscalar.nii \
  -cifti \
  ${data_dir}/norepinephrine/ADRA2A.pscalar.nii \
  -cifti \
  ${data_dir}/norepinephrine/ADRA2C.pscalar.nii \
  -cifti \
  ${data_dir}/norepinephrine/ADRB1.pscalar.nii \
  -cifti \
  ${data_dir}/norepinephrine/ADRB2.pscalar.nii 