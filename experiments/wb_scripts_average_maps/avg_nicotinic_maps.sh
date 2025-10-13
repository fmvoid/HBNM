data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

wb_command -cifti-average \
  ${data_dir}/nicotinic_avg.pscalar.nii \
  -cifti \
  ${data_dir}/nicotinic/CHRNA2.pscalar.nii \
  -cifti \
  ${data_dir}/nicotinic/CHRNA3.pscalar.nii \
  -cifti \
  ${data_dir}/nicotinic/CHRNA4.pscalar.nii \
  -cifti \
  ${data_dir}/nicotinic/CHRNA7.pscalar.nii \
  -cifti \
  ${data_dir}/nicotinic/CHRNA10.pscalar.nii \
  -cifti \
  ${data_dir}/nicotinic/CHRNB1.pscalar.nii \
  -cifti \
  ${data_dir}/nicotinic/CHRNB2.pscalar.nii 