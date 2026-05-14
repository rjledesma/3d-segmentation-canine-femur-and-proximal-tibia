https://zenodo.org/records/10047292
Dataset

----------------------------------
# Citations

# Itk
If you publish results obtained using ITK-SNAP, please cite the following paper:

    Paul A. Yushkevich, Joseph Piven, Heather Cody Hazlett, Rachel Gimpel Smith, Sean Ho, James C. Gee, and Guido Gerig. User-guided 3D active contour segmentation of anatomical structures: Significantly improved efficiency and reliability. Neuroimage. 2006 Jul 1; 31(3):1116-28.
    [bibtex] [medline] [doi:10.1016/j.neuroimage.2006.01.015] 

# Total Segmentator
If you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024

----------------------------------
# Dice Scores from training

80 patients femur only: Dice Scores ~0.97-0.99

----------------------------------
# shell scripts
# nnunet predict "-chk checkpoint_best.pth for best path" "-chk checkpoint_latest for latest path"
nnUNetv2_predict `
-i "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\predict_input" `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\predict_output" `
-d 1 `
-c 3d_fullres `
-f 0 `
-chk checkpoint_best.pth

# Predict femur trained from old dataset on VSD dataset
nnUNetv2_predict `
-i "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_predict_input" `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_femur_pred" `
-d 1 `
-c 3d_fullres `
-f 0 `
-chk checkpoint_best.pth

# total segmentator generate dataset
TotalSegmentator `
-i "C:\Users\rjled\Downloads\TotalSegmentator_dataset\s0000\ct.nii.gz" `
-o "C:\Users\rjled\Downloads\TotalSegmentator_dataset\s0000\appendicular_bones" `
--task appendicular_bones

# preprocess
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity --clean

# paths to setup before using nnunet
$env:nnUNet_raw="C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_raw"
$env:nnUNet_preprocessed="C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_preprocessed"
$env:nnUNet_results="C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_results"

# TotalSeg on Lower extremities for tibia and femur dcm2niix
TotalSegmentator `
-i "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti_dcm2niix\vsd_001_lower_limb_2.nii.gz" `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg\vsd_001_lower_limb_2_appendicular" `
--task appendicular_bones

# Single threaded version lower extremities on dcm2niix
TotalSegmentator `
-i "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti_dcm2niix\vsd_001_lower_limb_2.nii.gz" `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg\vsd_001_lower_limb_2_appendicular" `
--task appendicular_bones `
--nr_thr_resamp 1 `
--nr_thr_saving 1

# Dicom to Nifti conversion
dcm2niix `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti_dcm2niix" `
-f "vsd_001_lower_limb_2" `
"C:\Users\rjled\Downloads\VSDFullBodyDataset\001\SMIR.Lower_limb.089Y.M.CT.2"

# Gzip Compression
dcm2niix `
-z y `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti_dcm2niix" `
-f "vsd_001_lower_limb_2" `
"C:\Users\rjled\Downloads\VSDFullBodyDataset\001\SMIR.Lower_limb.089Y.M.CT.2"

# On simple itk nifti total segmentator for lower extremities
TotalSegmentator `
-i "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti\vsd_001_0001_0000.nii.gz" `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg\vsd_001_0001_appendicular" `
--task appendicular_bones

----------------------------------
# Notes

# Dataset used
The project will use the TotalSegmentator CT dataset v2.0.1 as the primary dataset source. It contains 1228 CT volumes with anatomical segmentations. Since the task targets femur and tibia, all cases will be screened for target anatomy. Cases with femur/tibia presence will be extracted into nnU-Net format. For tibia labels not present in the default segmentation folder, TotalSegmentator’s appendicular-bone task will be used to generate tibia masks, followed by quality checking.

# Total Segmentator
https://zenodo.org/records/10047292

# Full Body CT VSD
https://zenodo.org/records/8270365


For each tibia mask:
Find the bounding box of the full tibia.
Determine the superior/inferior axis.
Keep only the proximal portion, for example the upper 25–30% of the tibia length.
Save that as class 2.

Dataset001_HumanFemur as baseline evidence:
| Model                 | Dataset    | Target                 |
| --------------------- | ---------- | ---------------------- |
| Baseline              | Dataset001 | Femur only             |
| Improved thesis model | Dataset002 | Femur + Proximal Tibia |

# How the total segmentator dataset was screened
The full TotalSegmentator v2.0.1 dataset containing 1228 CT volumes was screened. From these, 93 lower-extremity candidate scans were identified using metadata, and cases with valid femur and proximal tibia masks were included.

----------------------------------
# Training
70% train
15% validation
15% test

# Updated NnuNet to use 300 epochs instead of 1000 on default:
C:\vscode_windows\thesis\3d-segmentation-femur-tibia\venv310\Lib\site-packages\nnunetv2
created new class on its package
nnUNetTrainer_300epochs.py
can be used using "nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_300epochs"
to resume use this params "nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_300epochs --c"

# Mean Validation Dice on VSDFullBody
2026-05-12 12:01:36.308171: Mean Validation Dice:  0.8029940765249004
Training done.
This split has 28 training and 7 validation cases.
Validation complete
Mean Validation Dice: 0.8029940765249004

----------------------------------


# APP STRUCTURE
Top Bar
Femur-Tibia 3D Segmentation System        Status: Ready

Left Panel
Upload CT Scan
Patient / Case Info
Model Info
Segmentation Progress

Center Panel
3D Output Viewer
Femur + Proximal Tibia model

Right Panel
Input CT Slices
Prediction Mask Preview
Export Results

----------------------------------


# TO DOS
So the plan is:

Keep Dataset001_HumanFemur as baseline.
Build Dataset002_HumanFemurProximalTibia.
Train a new 300-epoch model on Dataset002.
Compare baseline vs improved model in your thesis.

----------------------------------

# TESTS

# Backend testing
http://127.0.0.1:8000/docs

# Front End
python -m http.server 5500

# Server
uvicorn app.main:app --reload

----------------------------------
# Key Observations
Larger DICOM CT cases takes too long to segment.
Example:
Dicom ZIP with 1944 slices, using my GPU RTX 4070 takes 2.8 sec per slice, 1944 × 2.8 sec ≈ 90 minutes
Solution:
using crop_service.py automatic cropping
DICOM ZIP
→ Convert to NIfTI
→ Crop lower-limb/knee ROI
→ Run nnU-Net on cropped NIfTI
→ Convert mask to STL