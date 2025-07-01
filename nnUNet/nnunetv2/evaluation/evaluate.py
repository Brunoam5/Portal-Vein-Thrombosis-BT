import sys
import os
from pathlib import Path
import shutil
from functools import partial

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_map
# package from: https://github.com/deepmind/surface-distance
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance  


def dice_score(y_true, y_pred):
    """
    Binary Dice score. Same results as sklearn f1 binary.
    """
    intersect = np.sum(y_true * y_pred)  
    denominator = np.sum(y_true) + np.sum(y_pred)
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def calc_metrics(subject, gt_dir=None, pred_dir=None, class_map=None):
    gt_all = nib.load(gt_dir / f"{subject}.nii.gz").get_fdata()
    pred_all = nib.load(pred_dir / f"{subject}.nii.gz").get_fdata()

    r = {"subject": subject}
    for idx, roi_name in class_map.items():
        gt = gt_all == idx
        pred = pred_all == idx
    
        if gt.max() > 0 and pred.max() == 0:
            r[f"dice-{roi_name}"] = 0
            r[f"surface_dice_3-{roi_name}"] = 0
        elif gt.max() > 0:
            r[f"dice-{roi_name}"] = dice_score(gt, pred)
            sd = compute_surface_distances(gt, pred, [1.5, 1.5, 1.5])
            r[f"surface_dice_3-{roi_name}"] = compute_surface_dice_at_tolerance(sd, 3.0)
        # gt.max() == 0 which means we can not calculate any score because roi not in the image
        else:  
            r[f"dice-{roi_name}"] = np.NaN
            r[f"surface_dice_3-{roi_name}"] = np.NaN
        print(f"ROI {roi_name} - gt max: {gt.max()}, pred max: {pred.max()}")
    return r


if __name__ == "__main__":
    """
    Calculate Dice score and normalized surface distance for your nnU-Net predictions.

    example usage: 
    python evaluate.py ground_truth_dir predictions_dir
    """
    gt_dir = Path(sys.argv[1])  # directory containining all the subjects
    pred_dir = Path(sys.argv[2])  # directory of the new nnunet dataset

    # Manually define the class map
    class_map = {1: "thrombus"}

    subjects = [x.stem.split(".")[0] for x in gt_dir.glob("*.nii.gz")]
    print("Subjects to process:", subjects)

    # Use multiple threads to calculate the metrics
    res = p_map(partial(calc_metrics, gt_dir=gt_dir, pred_dir=pred_dir,
                        class_map=class_map), subjects, num_cpus=8, disable=True)
    res = pd.DataFrame(res)
    print("\n### Individual Dice Scores ###")
    for metric in ["dice", "surface_dice_3"]:
        for roi_name in class_map.values():
            print(f"\n{metric} scores for {roi_name}:")
            for _, row in res.iterrows():
                print(f"Subject {row['subject']}: {row[f'{metric}-{roi_name}']:.3f}")
    #print("Columns in res:", res.columns)
    #for metric in ["dice", "surface_dice_3"]:
        #res_all_rois = []
        #for roi_name in class_map.values():
            #row_wo_nan = res[f"{metric}-{roi_name}"].dropna()
            #res_all_rois.append(row_wo_nan.mean())
            #print(f"{roi_name} {metric}: {row_wo_nan.mean():.3f}")