# makesure to use activate venv first
$env:nnUNet_raw = "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_preprocessed"
$env:nnUNet_results = "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_results"

$INPUT_DIR = "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\predict_input"
$OUTPUT_DIR = "C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\predict_output"

nnUNetv2_predict `
    -i $INPUT_DIR `
    -o $OUTPUT_DIR `
    -d 1 `
    -c 3d_fullres `
    -f 0 `
    -chk checkpoint_best.pth