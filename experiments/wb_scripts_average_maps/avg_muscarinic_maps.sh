data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

wb_command -cifti-average \
  ${data_dir}/muscarinic_avg.pscalar.nii \
  -cifti \
  ${data_dir}/muscarinic/CHRM1.pscalar.nii \
  -cifti \
  ${data_dir}/muscarinic/CHRM2.pscalar.nii \
  -cifti \
  ${data_dir}/muscarinic/CHRM3.pscalar.nii \
  -cifti \
  ${data_dir}/muscarinic/CHRM4.pscalar.nii \
  -cifti \
  ${data_dir}/muscarinic/CHRM5.pscalar.nii 