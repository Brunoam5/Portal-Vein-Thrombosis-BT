import SimpleITK as sitk
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, isfile
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager



import numpy as np
import nibabel as nib
import json
from batchgenerators.utilities.file_and_folder_operations import subfiles, save_json
from scipy.ndimage import zoom

def resample_to_match(img_ref, img_pred):
    """Resamples the reference image to match the prediction's shape and spacing."""
    ref_data = img_ref.get_fdata()
    pred_data = img_pred.get_fdata()

    # Get the spacing
    ref_spacing = img_ref.header.get_zooms()
    pred_spacing = img_pred.header.get_zooms()

    # Compute new shape to match prediction
    scale_factors = np.array(ref_spacing) / np.array(pred_spacing)
    new_shape = np.round(np.array(ref_data.shape) * scale_factors).astype(int)

    # Resample reference image
    resampled_ref = zoom(ref_data, scale_factors, order=1)  # Linear interpolation

    # Create new nibabel image with original header
    new_img = nib.Nifti1Image(resampled_ref, img_pred.affine, img_pred.header)
    
    return new_img

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def check_and_resample_gt(gt_path, pred_path):
    """Loads GT and Prediction, checks misalignment, and resamples if needed."""
    
    # Load images with SimpleITK
    gt_sitk = sitk.ReadImage(gt_path)  # FIXED: Removed SetFileNames
    pred_sitk = sitk.ReadImage(pred_path)

    # Convert to numpy
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    pred_data = sitk.GetArrayFromImage(pred_sitk)

    # Get spacings
    gt_spacing = np.array(gt_sitk.GetSpacing())[::-1]  # Reverse for numpy indexing
    pred_spacing = np.array(pred_sitk.GetSpacing())[::-1]

    # Compute resampling factor
    scale_factors = gt_spacing / pred_spacing
    new_shape = np.round(gt_data.shape * scale_factors).astype(int)

    # Resample GT to match prediction
    gt_resampled = zoom(gt_data, scale_factors, order=1)

    # Save resampled GT
    new_gt_sitk = sitk.GetImageFromArray(gt_resampled)
    new_gt_sitk.SetSpacing(pred_sitk.GetSpacing())  # Match spacing
    new_gt_sitk.SetOrigin(pred_sitk.GetOrigin())  # Match origin

    resampled_gt_path = gt_path.replace(".nii.gz", "_resampled.nii.gz")
    sitk.WriteImage(new_gt_sitk, resampled_gt_path)

    return resampled_gt_path

def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions, ignore_label=None):
    """Compute Dice, IoU and other metrics after ensuring GT and prediction are aligned."""
    # Ensure GT is aligned with prediction
    check_and_resample_gt(reference_file, prediction_file)

    # Load images after resampling
    seg_ref, _ = image_reader_writer.read_seg(reference_file)
    seg_pred, _ = image_reader_writer.read_seg(prediction_file)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {'reference_file': reference_file, 'prediction_file': prediction_file, 'metrics': {}}

    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = seg_ref == r
        mask_pred = seg_pred == r
        tp = np.sum((mask_ref & mask_pred) & ~ignore_mask) if ignore_mask is not None else np.sum(mask_ref & mask_pred)
        fp = np.sum((~mask_ref & mask_pred) & ~ignore_mask) if ignore_mask is not None else np.sum(~mask_ref & mask_pred)
        fn = np.sum((mask_ref & ~mask_pred) & ~ignore_mask) if ignore_mask is not None else np.sum(mask_ref & ~mask_pred)

        dice = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

        results['metrics'][r]['Dice'] = dice
        results['metrics'][r]['IoU'] = iou
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
    return results


import numpy as np

def compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, labels_or_regions, ignore_label=None, num_processes=8):
    """Compute metrics on all GT-prediction pairs in a folder, ensuring GT is aligned."""
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)

    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    results = []
    for ref_path, pred_path in zip(files_ref, files_pred):
        img_ref = nib.load(ref_path)
        img_pred = nib.load(pred_path)

        # Resample reference to match prediction
        img_ref = resample_to_match(img_ref, img_pred)

        metrics = compute_metrics(img_ref.get_fdata(), img_pred.get_fdata(), labels_or_regions, ignore_label)
        results.append({
            "metrics": {str(k): {metric: float(v) for metric, v in metrics[k].items()} for k in metrics},
            "reference_file": ref_path,
            "prediction_file": pred_path
        })

    mean_metrics = {
        str(r): {
            'Dice': float(np.nanmean([res['metrics'][str(r)]['Dice'] for res in results])),
            'IoU': float(np.nanmean([res['metrics'][str(r)]['IoU'] for res in results]))
        } for r in labels_or_regions
    }

    summary = {'metric_per_case': results, 'mean': mean_metrics}
    save_json(summary, output_file)

    return summary


# Example Usage
if __name__ == '__main__':
    folder_ref = '/gpfs/home/bandres/nnUNetFrame/dataset/nnUNet_raw/Dataset003_ThrombusDataset/labelsTs'
    folder_pred = '/gpfs/home/bandres/nnUNetFrame/dataset/nnUNet_raw/Dataset003_ThrombusDataset/predictedTs_e4'
    output_file = '/gpfs/home/bandres/nnUNetFrame/dataset/nnUNet_raw/Dataset003_ThrombusDataset/summary.json'
    
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = [(1,), (2,)]  # Labels to evaluate
    ignore_label = None
    num_processes = 12

    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label, num_processes)
