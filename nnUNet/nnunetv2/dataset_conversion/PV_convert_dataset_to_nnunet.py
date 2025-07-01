import sys
import os
from pathlib import Path
import shutil
import json
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm


def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, labels):
    print("Creating dataset.json...")
    out_base = foldername
    out_base.mkdir(parents=True, exist_ok=True)

    json_dict = {
        'name': "Own_Dataset_PortalVein_PVT",
        'description': "Segmentation of own cases of portal vein thrombosis",
        'channel_names': {"0": "CT"},
        'labels': {"background": 0, "portal_vein_and_splenic_vein": 1, "thrombus": 2},  # Only portal vein included in labels
        'numTraining': len(subjects_train) + len(subjects_val),
        'file_ending': '.nii.gz',
        'overwrite_image_reader_writer': 'NibabelIOWithReorient'
    }

    json.dump(json_dict, open(out_base / "dataset.json", "w"), sort_keys=False, indent=4)

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnUNet_preprocessed']) / foldername.name
    print(output_folder_pkl)
    output_folder_pkl.mkdir(parents=True, exist_ok=True)

    splits = [{"train": subjects_train, "val": subjects_val}]

    print(f"nr of folds: {len(splits)}")
    print(f"nr train subjects (fold 0): {len(splits[0]['train'])}")
    print(f"nr val subjects (fold 0): {len(splits[0]['val'])}")

    json.dump(splits, open(output_folder_pkl / "splits_final.json", "w"), sort_keys=False, indent=4)


def combine_labels(ref_img, file_out, masks, labels):
    ref_img = nib.load(ref_img)
    combined = np.zeros(ref_img.shape, dtype=np.uint8)

    # Only process the 'portal_vein' label mask
    portal_vein_mask = Path(masks / "portal_vein.nii.gz")
    
    if portal_vein_mask.exists():
        img = nib.load(portal_vein_mask)
        combined[img.get_fdata() > 0] = 1  # Set portal vein mask to label 1
    else:
        print(f"Missing: {portal_vein_mask}")

    # Save the combined mask with only portal vein
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


if __name__ == "__main__":
    dataset_path = Path("/gpfs/home/bandres/nnUNetDataset/Inhouse_database")
    nnunet_path = Path("/gpfs/home/bandres/nnUNetFrame/dataset/nnUNet_raw/Dataset002_PVTDataset")

    labels = {
        "background": 0,
        "portal_vein_and_splenic_vein":1,
        "thrombus": 2
    }

    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(dataset_path / "meta.csv", sep=",")
    subjects_train = list(meta[meta["split"] == "train"]["image_id"].values)
    subjects_val = list(meta[meta["split"] == "val"]["image_id"].values)
    subjects_test = list(meta[meta["split"] == "test"]["image_id"].values)

    print("Copying train data...")
    
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / subject
        print(f"Processing patient: {subject}")
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTr" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
               nnunet_path / "labelsTr" / f"{subject}.nii.gz",
               subject_path / "segmentations", labels)

    print("Copying test data...")
    for subject in tqdm(subjects_test):
        subject_path = dataset_path / subject
        print(f"Processing patient: {subject}")
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
                       nnunet_path / "labelsTs" / f"{subject}.nii.gz",
                       subject_path / "segmentations", labels)

    generate_json_from_dir_v2(nnunet_path, subjects_train, subjects_val, labels)