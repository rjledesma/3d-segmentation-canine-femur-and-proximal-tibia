from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

HUMAN_DICOM_DIR = DATA_DIR / "source_human" / "dicom"
HUMAN_NIFTI_DIR = DATA_DIR / "source_human" / "nifti"
HUMAN_LABELS_DIR = DATA_DIR / "source_human" / "labels"

CANINE_DICOM_DIR = DATA_DIR / "source_canine" / "dicom"
CANINE_NIFTI_DIR = DATA_DIR / "source_canine" / "nifti"
CANINE_LABELS_DIR = DATA_DIR / "source_canine" / "labels"

NNUNET_RAW_DIR = DATA_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED_DIR = DATA_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS_DIR = DATA_DIR / "nnUNet_results"

HUMAN_DATASET_NAME = "Dataset001_HumanFemurTibia"
CANINE_DATASET_NAME = "Dataset002_CanineFemurTibia"

HUMAN_DATASET_DIR = NNUNET_RAW_DIR / HUMAN_DATASET_NAME
CANINE_DATASET_DIR = NNUNET_RAW_DIR / CANINE_DATASET_NAME