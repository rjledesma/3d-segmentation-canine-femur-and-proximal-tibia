from pathlib import Path
import nibabel as nib
import numpy as np

TOTALSEG_ROOT = Path(r"C:\Users\rjled\Downloads\TotalSegmentator_dataset")

NNUNET_DATASET_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_raw\Dataset001_HumanFemur"
)

IMAGES_TR = NNUNET_DATASET_DIR / "imagesTr"
LABELS_TR = NNUNET_DATASET_DIR / "labelsTr"

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)


def load_nifti(path: Path):
    return nib.load(str(path))


def save_nifti(data, affine, header, out_path: Path):
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(out_path))


def process_case(case_dir: Path, case_index: int):
    ct_path = case_dir / "ct.nii.gz"
    seg_dir = case_dir / "segmentations"

    femur_left_path = seg_dir / "femur_left.nii.gz"
    femur_right_path = seg_dir / "femur_right.nii.gz"

    if not ct_path.exists():
        print(f"[SKIP] Missing CT: {case_dir.name}")
        return False

    if not femur_left_path.exists() and not femur_right_path.exists():
        print(f"[SKIP] No femur masks: {case_dir.name}")
        return False

    ct_img = load_nifti(ct_path)
    ct_data = ct_img.get_fdata().astype(np.float32)

    label = np.zeros(ct_data.shape, dtype=np.uint8)

    if femur_left_path.exists():
        femur_left = load_nifti(femur_left_path).get_fdata()
        label[femur_left > 0] = 1

    if femur_right_path.exists():
        femur_right = load_nifti(femur_right_path).get_fdata()
        label[femur_right > 0] = 1

    case_id = f"human_{case_index:04d}"

    out_image = IMAGES_TR / f"{case_id}_0000.nii.gz"
    out_label = LABELS_TR / f"{case_id}.nii.gz"

    # Create clean NIfTI image for CT
    ct_out = nib.Nifti1Image(ct_data, ct_img.affine)
    ct_out.header.set_data_dtype(np.float32)
    ct_out.set_qform(ct_img.affine, code=1)
    ct_out.set_sform(ct_img.affine, code=1)

    # Create clean NIfTI image for label
    label_out = nib.Nifti1Image(label.astype(np.uint8), ct_img.affine)
    label_out.header.set_data_dtype(np.uint8)
    label_out.set_qform(ct_img.affine, code=1)
    label_out.set_sform(ct_img.affine, code=1)

    # Save both
    nib.save(ct_out, str(out_image))
    nib.save(label_out, str(out_label))

    print(f"[OK] {case_dir.name} -> {case_id}")
    return True


def main():
    case_dirs = sorted(
        [p for p in TOTALSEG_ROOT.iterdir() if p.is_dir() and p.name.startswith("s")]
    )

    saved_count = 0
    case_index = 1

    # 10 cases first
    for case_dir in case_dirs[:10]:
        success = process_case(case_dir, case_index)
        if success:
            saved_count += 1
            case_index += 1

    print(f"\nDone. Saved {saved_count} cases.")


if __name__ == "__main__":
    main()