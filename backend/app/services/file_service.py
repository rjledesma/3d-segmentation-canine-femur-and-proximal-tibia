from pathlib import Path
import shutil
import uuid


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def create_case_id() -> str:
    return f"case_{uuid.uuid4().hex[:8]}"


def create_case_dirs(case_id: str) -> dict:
    case_dir = PROCESSED_DIR / case_id
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    mesh_dir = case_dir / "meshes"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    return {
        "case_dir": case_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "mesh_dir": mesh_dir,
    }


def save_upload_file(upload_file, destination: Path) -> None:
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def normalize_nnunet_input_name(original_filename: str, input_dir: Path) -> Path:
    """
    nnU-Net expects input files as:
    case_0000.nii.gz

    For NIfTI, this backend stores the uploaded file as:
    input_0000.nii.gz or input_0000.nii

    For ZIP, the uploaded ZIP is stored temporarily as:
    upload.zip
    then converted to input_0000.nii.gz
    """
    lower = original_filename.lower()

    if lower.endswith(".nii.gz"):
        return input_dir / "input_0000.nii.gz"

    if lower.endswith(".nii"):
        return input_dir / "input_0000.nii"

    if lower.endswith(".zip"):
        return input_dir / "upload.zip"

    raise ValueError("Only .nii, .nii.gz, and .zip files are supported.")