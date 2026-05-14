$env:nnUNet_raw="C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_raw"
$env:nnUNet_preprocessed="C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_preprocessed"
$env:nnUNet_results="C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_results"

nnUNetv2_predict `
-i "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\dataset002_test_input" `
-o "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\dataset002_test_output" `
-d 2 `
-c 3d_fullres `
-tr nnUNetTrainer_300epochs `
-f 0 `
-chk checkpoint_best.pth