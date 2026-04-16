from pathlib import Path
import SimpleITK as sitk

DICOM_ROOT = Path(r"C:\Users\rjled\OneDrive\Documents\manifest-1771432770117\CT Lymph Nodes")
OUTPUT_DIR = Path(r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\predict_input")

TARGET_CASES = 5 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_dicom_series_dirs(root: Path):
    """
    Recursively find folders that contain DICOM files
    """
    dicom_dirs = []

    for p in root.rglob("*"):
        if p.is_dir():
            dcm_files = list(p.glob("*.dcm"))
            if len(dcm_files) > 10:
                dicom_dirs.append(p)

    return sorted(dicom_dirs)


def convert_dicom_to_nifti(dicom_dir: Path, out_path: Path):
    try:
        reader = sitk.ImageSeriesReader()

        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not series_ids:
            print(f"[SKIP] No series found: {dicom_dir}")
            return False

        series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(series_files)

        image = reader.Execute()

        sitk.WriteImage(image, str(out_path))

        print(f"[OK] {dicom_dir.name} -> {out_path.name}")
        return True

    except Exception as e:
        print(f"[SKIP] Failed {dicom_dir.name}: {e}")
        return False


def main():
    dicom_dirs = find_dicom_series_dirs(DICOM_ROOT)

    print(f"Found {len(dicom_dirs)} possible DICOM series")

    saved = 0
    case_index = 1

    for dicom_dir in dicom_dirs:
        case_id = f"human_test_{case_index:04d}_0000.nii.gz"
        out_path = OUTPUT_DIR / case_id

        success = convert_dicom_to_nifti(dicom_dir, out_path)

        if success:
            saved += 1
            case_index += 1

            if saved >= TARGET_CASES:
                break

    print(f"\nDone. Converted {saved} cases.")


if __name__ == "__main__":
    main()