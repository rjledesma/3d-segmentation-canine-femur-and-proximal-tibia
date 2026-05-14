from pathlib import Path
import zipfile
import shutil
import SimpleITK as sitk


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """
    Extracts uploaded DICOM ZIP into extract_dir.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    return extract_dir


def find_dicom_series(root_dir: Path):
    """
    Finds all DICOM series inside an extracted folder.
    Returns list of (series_dir, series_id, dicom_files).
    """
    reader = sitk.ImageSeriesReader()
    series_results = []

    candidate_dirs = set()

    for file_path in root_dir.rglob("*"):
        if file_path.is_file():
            candidate_dirs.add(file_path.parent)

    for folder in sorted(candidate_dirs):
        try:
            series_ids = reader.GetGDCMSeriesIDs(str(folder))
        except RuntimeError:
            continue

        if not series_ids:
            continue

        for series_id in series_ids:
            try:
                dicom_files = reader.GetGDCMSeriesFileNames(str(folder), series_id)
            except RuntimeError:
                continue

            if len(dicom_files) > 0:
                series_results.append((folder, series_id, dicom_files))

    return series_results


def convert_largest_dicom_series_to_nifti(
    dicom_root: Path,
    output_nifti_path: Path,
) -> Path:
    """
    Finds the largest DICOM series inside dicom_root and converts it to NIfTI.
    Output should be named input_0000.nii.gz for nnU-Net.
    """
    series_results = find_dicom_series(dicom_root)

    if not series_results:
        raise ValueError(f"No valid DICOM series found in: {dicom_root}")

    # Pick largest series by slice count
    series_results.sort(key=lambda x: len(x[2]), reverse=True)
    series_dir, series_id, dicom_files = series_results[0]

    print("[DICOM] Selected series:")
    print(f"  folder: {series_dir}")
    print(f"  series_id: {series_id}")
    print(f"  slices: {len(dicom_files)}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)

    image = reader.Execute()

    output_nifti_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_nifti_path))

    print(f"[DICOM] Saved NIfTI: {output_nifti_path}")

    return output_nifti_path


def prepare_zip_as_nnunet_input(
    zip_path: Path,
    case_dir: Path,
    input_dir: Path,
) -> Path:
    """
    Extracts DICOM ZIP and converts largest DICOM series to:
    input/input_0000.nii.gz
    """
    extracted_dir = case_dir / "dicom_extracted"

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    extract_zip(zip_path, extracted_dir)

    output_nifti_path = input_dir / "input_0000.nii.gz"

    return convert_largest_dicom_series_to_nifti(
        dicom_root=extracted_dir,
        output_nifti_path=output_nifti_path,
    )