from pathlib import Path
import os
import subprocess


NNUNET_DATA_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data"
)

NNUNET_RAW = NNUNET_DATA_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_DATA_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_DATA_DIR / "nnUNet_results"


def run_nnunet_prediction(input_dir: Path, output_dir: Path) -> Path:
    """
    Runs Dataset002_HumanFemurProximalTibia prediction.

    Expected output:
    output/input.nii.gz
    """
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)

    cmd = [
    "nnUNetv2_predict",
    "-i", str(input_dir),
    "-o", str(output_dir),
    "-d", "2",
    "-c", "3d_fullres",
    "-tr", "nnUNetTrainer_300epochs",
    "-f", "0",
    "-chk", "checkpoint_best.pth",
    "--disable_tta",
    "-npp", "1",
    "-nps", "1",
    ]

    print("[RUN]", " ".join(cmd))

    result = subprocess.run(
        cmd,
        env=env,
    )

    if result.returncode != 0:
        print("[STDOUT]", result.stdout)
        print("[STDERR]", result.stderr)
        raise RuntimeError(
            "nnU-Net prediction failed.\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    prediction_path = output_dir / "input.nii.gz"

    if not prediction_path.exists():
        # fallback: find any .nii.gz output
        outputs = list(output_dir.glob("*.nii.gz"))
        if not outputs:
            raise FileNotFoundError(f"No prediction output found in {output_dir}")
        prediction_path = outputs[0]

    return prediction_path