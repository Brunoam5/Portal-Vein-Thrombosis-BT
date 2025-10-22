import os
import subprocess
import pydicom

def custom_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_valid_dicom_image(dicom_folder):
    for root, _, files in os.walk(dicom_folder):
        for f in files:
            if f.lower().endswith(".dcm") or '.' not in f:
                try:
                    ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                    modality = ds.get("Modality", "")
                    if modality in ["CT"]:
                        return True
                except:
                    continue
    return False

def dcm2nii(dicom_path, nifti_path, folder_name):
    cmd = f"dcm2niix -z y -f {folder_name} -o {nifti_path} {dicom_path}"
    os.system(cmd)

def main(dicom_root_path, nifti_root_path):
    custom_mkdir(nifti_root_path)

    for root, dirs, files in os.walk(dicom_root_path):
        if any(fname.lower().endswith('.dcm') or not '.' in fname for fname in files):
            if not is_valid_dicom_image(root):
                print(f"No es una imagen CT/MR válida: {root}")
                continue

            parts = root.split(os.sep)
            patient_id = next((p for p in parts if p.startswith("PAC")), "UNKNOWN")
            output_path = os.path.join(nifti_root_path, patient_id)
            folder_name = f"{patient_id}_{parts[-1]}"

            # Verifica si el NIfTI ya existe
            if os.path.exists(output_path):
                existing_nii = any(f.startswith(folder_name) and (f.endswith(".nii") or f.endswith(".nii.gz"))
                                   for f in os.listdir(output_path))
                if existing_nii:
                    print(f"Ya existe NIfTI para {patient_id}. Saltando...")
                    continue

            custom_mkdir(output_path)
            print(f"Convirtiendo {root} -> {output_path}/{folder_name}.nii")
            dcm2nii(root, output_path, folder_name)

if __name__ == '__main__':
    dicom_root_path = "Introduce path to dicom file"
    nifti_root_path = "Introduce path to nifti file"
    main(dicom_root_path, nifti_root_path)
