import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Paths
absolute_path = "Introduce path to Database"
out_path_slices = "Introduce output path for CT slices"
out_path_thrombus = "Introduce output path for Thrombus mask"
os.makedirs(out_path_slices, exist_ok=True)
os.makedirs(out_path_thrombus, exist_ok=True)

# Compute optimal spacing (mean of all the available spacings in the dataset)
def get_all_spacings(base_path): 
    spacings = []
    for patient in os.listdir(base_path):
        if patient.startswith("PAC"):
            patient_number = patient.split("PAC")[1]
            ct_path = os.path.join(base_path, patient, f'PAC_{patient_number}.nii')
            if os.path.exists(ct_path):
                try:
                    img = sitk.ReadImage(ct_path)
                    spacings.append(img.GetSpacing())
                except Exception as e:
                    print(f"❌ Error reading {ct_path}: {e}")
    return np.array(spacings)

spacings = get_all_spacings(absolute_path)

if spacings.size == 0:
    raise ValueError("No valid CT were found to compute mean spacing.")

mean_spacing = np.mean(spacings, axis=0)
new_spacing = [round(s, 1) for s in mean_spacing]  
print(f"Mean spacing applied: {new_spacing}")

# Parameters
new_spacing = [0.8, 0.8, 2.0] # Computed from the mean
padding = 10  # Voxels
window = (-100, 400)  # HU

def load_and_reorient(path):
    image = sitk.ReadImage(path)
    return sitk.DICOMOrient(image, 'LPS')

def resample_image(image, spacing):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if image.GetPixelID() == sitk.sitkUInt8:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)

def apply_window(image_array, window):
    return np.clip(image_array, *window)

def extract_bounding_box(mask_array, padding):
    coords = np.argwhere(mask_array > 0)
    min_coords = np.maximum(coords.min(axis=0) - padding, 0)
    max_coords = coords.max(axis=0) + padding + 1
    return tuple(slice(start, end) for start, end in zip(min_coords, max_coords))

def get_nib_affine(image_sitk):
    spacing = image_sitk.GetSpacing()
    direction = np.array(image_sitk.GetDirection()).reshape(3, 3)
    origin = np.array(image_sitk.GetOrigin())

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing  
    affine[:3, 3] = origin
    return affine

def process_patient(patient_id):
    patient_number = patient_id.split("PAC")[1]
    print(f"\n--- Processing PAC_{patient_number} ---")

    try:
        # Load and reorient
        image_sitk = load_and_reorient(os.path.join(absolute_path, patient_id, f'PAC_{patient_number}.nii'))
        mask_sitk = load_and_reorient(os.path.join(absolute_path, patient_id, f'PAC_{patient_number}_Thrombus.nii'))

        # Resampling to uniform spacing
        image_resampled = resample_image(image_sitk, new_spacing)
        mask_resampled = resample_image(mask_sitk, new_spacing)

        # Convert to arrays
        image = sitk.GetArrayFromImage(image_resampled)  # z, y, x
        mask = (sitk.GetArrayFromImage(mask_resampled) > 0).astype(np.uint8)

        if np.sum(mask) == 0:
            print("Empty mask. Skipping.")
            return

        # Apply windowing
        image = apply_window(image, window)

        # Extract bounding box
        slices = extract_bounding_box(mask, padding)
        image_crop = image[slices]
        mask_crop = mask[slices]

        # Save cropped volumes
        #affine = np.eye(4)  # Use of real affine instead
        affine = get_nib_affine(image_resampled)
        nib.save(nib.Nifti1Image(image_crop, affine), os.path.join(out_path_slices, f'PACS{patient_number}.nii'))
        nib.save(nib.Nifti1Image(mask_crop, affine), os.path.join(out_path_thrombus, f'PACS{patient_number}.nii'))

        print(f"PAC_{patient_number} processed: shape {image_crop.shape}, sum mask: {np.sum(mask_crop)}")

    except Exception as e:
        print(f"❌ Error en PAC_{patient_number}: {e}")

total = 0
for patient in os.listdir(absolute_path):
    if patient.startswith("PAC") and os.path.isdir(os.path.join(absolute_path, patient)):
        process_patient(patient)
        total += 1

print(f"\nAll the patients have been successfully processed. Total: {total}")

# Sanity check

def load_nifti(nifti_path):
    nii = nib.load(nifti_path)
    return nii.get_fdata()

def get_thrombus_center_slice(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    z_center = int(np.mean(coords[:, 2]))
    return z_center

def plot_multiple_slices(image, mask, center_slice, patient_id, range_slices=2):
    slices = range(center_slice - range_slices, center_slice + range_slices + 1)
    n_slices = len(slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 5))
    for i, z in enumerate(slices):
        if z < 0 or z >= image.shape[2]:
            axes[i].axis('off')
            continue
        axes[i].imshow(image[:, :, z], cmap='gray')
        axes[i].imshow(mask[:, :, z], cmap='Reds', alpha=0.5)
        axes[i].set_title(f"Slice {z}")
        axes[i].axis('off')
    
    plt.suptitle(f"Patient {patient_id} - Slices around thrombus center", fontsize=16)
    plt.tight_layout()
    plt.show()

patient_id = "PACSX"
image_path = os.path.join("path to CT", patient_id + ".nii")
mask_path = os.path.join("path to Thrombus mask", patient_id + ".nii")

if os.path.exists(image_path) and os.path.exists(mask_path):
    image = load_nifti(image_path)
    mask = load_nifti(mask_path)

    if image.shape != mask.shape:
        print(f"[ERROR] Shape mismatch for {patient_id}: image {image.shape}, mask {mask.shape}")
    else:
        center_slice = get_thrombus_center_slice(mask)
        if center_slice is None:
            print(f"[WARNING] Empty mask for patient {patient_id}")
        else:
            plot_multiple_slices(image, mask, center_slice, patient_id, range_slices=2)
else:
    print(f"[ERROR] Files from patient {patient_id} were not found")



