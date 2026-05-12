from pathlib import Path
import SimpleITK as sitk

VSD_ROOT = Path(r"C:\Users\rjled\Downloads\VSDFullBodyDataset")

OUT_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti"
)

TARGET_CASES = 50

OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_lower_limb_series(root: Path):
    series_dirs = []

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue

        # skip phantom unless you want it later
        if case_dir.name.lower().startswith("phantom"):
            continue

        for subdir in sorted(case_dir.iterdir()):
            if not subdir.is_dir():
                continue

            name = subdir.name.lower()

            if "lower_limb" not in name:
                continue

            dcm_count = len(list(subdir.glob("*.dcm")))
            if dcm_count < 20:
                continue

            series_dirs.append((case_dir.name, subdir, dcm_count))

    return series_dirs


def convert_series(dicom_dir: Path, out_path: Path):
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        print(f"[SKIP] No DICOM series found: {dicom_dir}")
        return False

    # Usually there is only one series per folder
    series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(series_files)

    try:
        image = reader.Execute()
        sitk.WriteImage(image, str(out_path))
        return True
    except Exception as e:
        print(f"[SKIP] Failed conversion: {dicom_dir}")
        print(e)
        return False


def main():
    lower_limb_dirs = find_lower_limb_series(VSD_ROOT)

    print(f"Found {len(lower_limb_dirs)} lower-limb DICOM series")

    saved = 0

    for case_name, dicom_dir, dcm_count in lower_limb_dirs:
        out_name = f"vsd_{case_name}_{saved + 1:04d}_0000.nii.gz"
        out_path = OUT_DIR / out_name

        print(f"\n[CONVERT] {case_name} | {dicom_dir.name} | {dcm_count} DICOM files")
        print(f"-> {out_path.name}")

        ok = convert_series(dicom_dir, out_path)

        if ok:
            saved += 1
            print(f"[OK] Saved {out_path}")

        if saved >= TARGET_CASES:
            break

    print(f"\nDone. Converted {saved} lower-limb cases.")


if __name__ == "__main__":
    main()