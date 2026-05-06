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

----------------------------------
# Notes

# Dataset used
The project will use the TotalSegmentator CT dataset v2.0.1 as the primary dataset source. It contains 1228 CT volumes with anatomical segmentations. Since the task targets femur and tibia, all cases will be screened for target anatomy. Cases with femur/tibia presence will be extracted into nnU-Net format. For tibia labels not present in the default segmentation folder, TotalSegmentator’s appendicular-bone task will be used to generate tibia masks, followed by quality checking.

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

