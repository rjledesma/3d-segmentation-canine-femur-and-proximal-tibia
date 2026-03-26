from pathlib import Path
from configs.paths import (
    HUMAN_DICOM_DIR,
    HUMAN_NIFTI_DIR,
    HUMAN_LABELS_DIR,
    CANINE_DICOM_DIR,
    CANINE_NIFTI_DIR,
    CANINE_LABELS_DIR,
    HUMAN_DATASET_DIR,
    CANINE_DATASET_DIR,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RESULTS_DIR,
)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {path}")

def main() -> None:
    dirs = [
        HUMAN_DICOM_DIR,
        HUMAN_NIFTI_DIR,
        HUMAN_LABELS_DIR,
        CANINE_DICOM_DIR,
        CANINE_NIFTI_DIR,
        CANINE_LABELS_DIR,
        HUMAN_DATASET_DIR / "imagesTr",
        HUMAN_DATASET_DIR / "labelsTr",
        CANINE_DATASET_DIR / "imagesTr",
        CANINE_DATASET_DIR / "labelsTr",
        NNUNET_PREPROCESSED_DIR,
        NNUNET_RESULTS_DIR,
    ]
    for d in dirs:
        ensure_dir(d)

if __name__ == "__main__":
    main()