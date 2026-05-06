from pathlib import Path
import nibabel as nib
import numpy as np

TOTALSEG_ROOT = Path(r"C:\Users\rjled\Downloads\TotalSegmentator_dataset")

NNUNET_DATASET_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_raw\Dataset002_HumanFemurProximalTibia"
)

IMAGES_TR = NNUNET_DATASET_DIR / "imagesTr"
LABELS_TR = NNUNET_DATASET_DIR / "labelsTr"

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)


def load_nifti(path: Path):
    return nib.load(str(path))


def is_non_empty(mask: np.ndarray) -> bool:
    return np.any(mask > 0)


def make_proximal_tibia(tibia_mask: np.ndarray, keep_ratio: float = 0.30) -> np.ndarray:
    """
    Creates a proximal tibia mask by keeping the superior 30% of the tibia bounding box.

    NOTE:
    This assumes the superior-inferior direction is approximately along the Z axis.
    You must visually check several outputs in 3D Slicer.
    """
    coords = np.argwhere(tibia_mask > 0)

    if coords.size == 0:
        return np.zeros_like(tibia_mask, dtype=np.uint8)

    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    z_len = z_max - z_min + 1

    keep_slices = max(1, int(z_len * keep_ratio))

    # Try superior side as higher Z first
    z_start = z_max - keep_slices + 1

    proximal = np.zeros_like(tibia_mask, dtype=np.uint8)
    proximal[:, :, z_start:z_max + 1] = tibia_mask[:, :, z_start:z_max + 1]

    return proximal.astype(np.uint8)


def process_case(case_dir: Path, case_index: int):
    ct_path = case_dir / "ct.nii.gz"

    seg_dir = case_dir / "segmentations"
    appendicular_dir = case_dir / "appendicular_bones"

    femur_left_path = seg_dir / "femur_left.nii.gz"
    femur_right_path = seg_dir / "femur_right.nii.gz"

    tibia_path = appendicular_dir / "tibia.nii.gz"

    if not ct_path.exists():
        print(f"[SKIP] Missing CT: {case_dir.name}")
        return False

    if not femur_left_path.exists() and not femur_right_path.exists():
        print(f"[SKIP] Missing femur masks: {case_dir.name}")
        return False

    if not tibia_path.exists():
        print(f"[SKIP] Missing tibia mask: {case_dir.name}")
        return False

    try:
        ct_img = load_nifti(ct_path)
        ct_data = ct_img.get_fdata().astype(np.float32)
    except Exception as e:
        print(f"[SKIP] Bad CT {case_dir.name}: {e}")
        return False

    label = np.zeros(ct_data.shape, dtype=np.uint8)

    # Femur = class 1
    try:
        if femur_left_path.exists():
            femur_left = load_nifti(femur_left_path).get_fdata()
            if femur_left.shape != ct_data.shape:
                print(f"[SKIP] femur_left shape mismatch: {case_dir.name}")
                return False
            label[femur_left > 0] = 1

        if femur_right_path.exists():
            femur_right = load_nifti(femur_right_path).get_fdata()
            if femur_right.shape != ct_data.shape:
                print(f"[SKIP] femur_right shape mismatch: {case_dir.name}")
                return False
            label[femur_right > 0] = 1

    except Exception as e:
        print(f"[SKIP] Bad femur mask {case_dir.name}: {e}")
        return False

    # Proximal tibia = class 2
    proximal_tibia_any = np.zeros(ct_data.shape, dtype=np.uint8)

    try:
        tibia = load_nifti(tibia_path).get_fdata()
        if tibia.shape != ct_data.shape:
            print(f"[SKIP] tibia shape mismatch: {case_dir.name}")
            return False

        proximal_tibia_any[make_proximal_tibia(tibia > 0) > 0] = 1

    except Exception as e:
        print(f"[SKIP] Bad tibia mask {case_dir.name}: {e}")
        return False

    if not np.any(label == 1):
        print(f"[SKIP] Empty femur label: {case_dir.name}")
        return False

    if not is_non_empty(proximal_tibia_any):
        print(f"[SKIP] Empty proximal tibia label: {case_dir.name}")
        return False

    label[proximal_tibia_any > 0] = 2

    case_id = f"human_{case_index:04d}"

    out_image = IMAGES_TR / f"{case_id}_0000.nii.gz"
    out_label = LABELS_TR / f"{case_id}.nii.gz"

    ct_out = nib.Nifti1Image(ct_data, ct_img.affine)
    ct_out.header.set_data_dtype(np.float32)
    ct_out.set_qform(ct_img.affine, code=1)
    ct_out.set_sform(ct_img.affine, code=1)

    label_out = nib.Nifti1Image(label.astype(np.uint8), ct_img.affine)
    label_out.header.set_data_dtype(np.uint8)
    label_out.set_qform(ct_img.affine, code=1)
    label_out.set_sform(ct_img.affine, code=1)

    nib.save(ct_out, str(out_image))
    nib.save(label_out, str(out_label))

    print(f"[OK] {case_dir.name} -> {case_id}")
    return True


def main():
    case_dirs = sorted(
        [p for p in TOTALSEG_ROOT.iterdir() if p.is_dir() and p.name.startswith("s")]
    )

    target_saved = 20
    saved_count = 1
    case_index = 1
    scanned_count = 0

    for case_dir in case_dirs:
        scanned_count += 1

        success = process_case(case_dir, case_index)

        if success:
            saved_count += 1
            case_index += 1

            if saved_count >= target_saved:
                break

    print(f"\nScanned {scanned_count} source cases.")
    print(f"Saved {saved_count} usable cases.")


if __name__ == "__main__":
    main()