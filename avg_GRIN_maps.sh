data_dir="/Users/fdjim/Desktop/HBNM/HBNM/data/maps"

wb_command -cifti-average \
  ${data_dir}/NMDA_avg.pscalar.nii \
  -cifti \
  ${data_dir}/GRIN1.pscalar.nii \
  -cifti \
  ${data_dir}/GRIN2A.pscalar.nii \
  -cifti \
  ${data_dir}/GRIN2B.pscalar.nii \
  -cifti \
  ${data_dir}/GRIN2C.pscalar.nii \
  -cifti \
  ${data_dir}/GRIN2D.pscalar.nii \
  -cifti \
  ${data_dir}/GRIN3A.pscalar.nii \
  -cifti \
  ${data_dir}/GRIN3B.pscalar.nii \
  -cifti \
  ${data_dir}/GRINA.pscalar.nii