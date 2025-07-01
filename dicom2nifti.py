import dicom2nifti
import nibabel as nib
import numpy as np
import os

path_patient = "Introduce path to dicom file"
path_out_data = "Introduce path for Nifti file output"

temp_nifti_path = os.path.join(path_out_data, "PAC_X_temp.nii")
dicom2nifti.dicom_series_to_nifti(path_patient, temp_nifti_path)
nifti_image = nib.load(temp_nifti_path)
image_array = nifti_image.get_fdata()
processed_nifti = nib.Nifti1Image(image_array, nifti_image.affine)
nib.save(processed_nifti, os.path.join(path_out_data, "PACX.nii"))
os.remove(temp_nifti_path)

print("Processing complete.")
