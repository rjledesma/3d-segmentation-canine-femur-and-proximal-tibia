from pathlib import Path
import shutil
import csv
import nibabel as nib
import numpy as np
from scipy import ndimage

# ===== PATHS =====
VSD_NIFTI_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti"
)

VSD_TOTALSEG_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg"
)

DATASET002_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_raw\Dataset002_HumanFemurProximalTibia"
)

IMAGES_TR = DATASET002_DIR / "imagesTr"
LABELS_TR = DATASET002_DIR / "labelsTr"

LOG_CSV = DATASET002_DIR / "dataset002_vsd_build_log.csv"

KEEP_RATIO = 0.30
MAX_CASES = 36  # use all currently usable VSD cases
# =================

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)


def load_nifti(path: Path):
    return nib.load(str(path))


def get_case_name_from_ct(ct_path: Path) -> str:
    return ct_path.name.replace("_0000.nii.gz", "")


def mask_voxels(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))


def make_proximal_tibia_near_femur(
    tibia_mask: np.ndarray,
    femur_mask: np.ndarray,
    keep_ratio: float = 0.30,
) -> np.ndarray:
    """
    Keeps the tibia end closest to the femur.
    This avoids relying on image orientation.
    """
    tibia_coords = np.argwhere(tibia_mask > 0)
    femur_coords = np.argwhere(femur_mask > 0)

    if tibia_coords.size == 0:
        return np.zeros_like(tibia_mask, dtype=np.uint8)

    if femur_coords.size == 0:
        raise ValueError("Femur mask is empty; cannot determine proximal tibia side.")

    z_min = tibia_coords[:, 2].min()
    z_max = tibia_coords[:, 2].max()
    z_len = z_max - z_min + 1

    keep_slices = max(1, int(z_len * keep_ratio))

    lower_end = np.zeros_like(tibia_mask, dtype=bool)
    upper_end = np.zeros_like(tibia_mask, dtype=bool)

    lower_end[:, :, z_min:z_min + keep_slices] = tibia_mask[:, :, z_min:z_min + keep_slices]
    upper_end[:, :, z_max - keep_slices + 1:z_max + 1] = tibia_mask[:, :, z_max - keep_slices + 1:z_max + 1]

    lower_coords = np.argwhere(lower_end > 0)
    upper_coords = np.argwhere(upper_end > 0)

    if lower_coords.size == 0 or upper_coords.size == 0:
        return np.zeros_like(tibia_mask, dtype=np.uint8)

    femur_center = femur_coords.mean(axis=0)
    lower_center = lower_coords.mean(axis=0)
    upper_center = upper_coords.mean(axis=0)

    lower_dist = np.linalg.norm(lower_center - femur_center)
    upper_dist = np.linalg.norm(upper_center - femur_center)

    if lower_dist < upper_dist:
        proximal = lower_end
        selected = "lower_z"
    else:
        proximal = upper_end
        selected = "upper_z"

    return proximal.astype(np.uint8), selected

def crop_to_mask_bbox(
    ct_data: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
    margin: int = 64,
):
    """
    Crop CT and label around the non-background label area.
    This reduces huge VSD volumes before nnU-Net preprocessing.
    """
    coords = np.argwhere(label > 0)

    if coords.size == 0:
        raise ValueError("Cannot crop because label is empty.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(label.shape))

    x0, y0, z0 = mins
    x1, y1, z1 = maxs

    cropped_ct = ct_data[x0:x1, y0:y1, z0:z1]
    cropped_label = label[x0:x1, y0:y1, z0:z1]

    # Update affine so physical coordinates remain correct after cropping
    new_affine = affine.copy()
    start_voxel = np.array([x0, y0, z0, 1])
    new_origin = affine @ start_voxel
    new_affine[:3, 3] = new_origin[:3]

    return cropped_ct, cropped_label, new_affine, (x0, x1, y0, y1, z0, z1)

def keep_largest_nonoverlapping_components(
    mask: np.ndarray,
    forbidden_mask: np.ndarray,
    max_components: int = 2,
    max_overlap_ratio: float = 0.05,
) -> np.ndarray:
    """
    Keeps the largest mask components that do not significantly overlap
    with the forbidden mask. For tibia, forbidden_mask should be femur.
    """
    labeled, num = ndimage.label(mask)

    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    components = []

    for label_id in range(1, num + 1):
        comp = labeled == label_id
        size = int(np.sum(comp))

        if size == 0:
            continue

        overlap = int(np.sum(comp & forbidden_mask))
        overlap_ratio = overlap / size

        if overlap_ratio <= max_overlap_ratio:
            components.append((size, label_id, overlap_ratio))

    if not components:
        return np.zeros_like(mask, dtype=bool)

    components.sort(reverse=True)

    keep_ids = [label_id for _, label_id, _ in components[:max_components]]

    cleaned = np.isin(labeled, keep_ids)

    return cleaned

def build_case(ct_path: Path, output_index: int):
    source_case = get_case_name_from_ct(ct_path)

    appendicular_dir = VSD_TOTALSEG_DIR / f"{source_case}_appendicular"
    femur_dir = VSD_TOTALSEG_DIR / f"{source_case}_femur"

    tibia_path = appendicular_dir / "tibia.nii.gz"
    femur_left_path = femur_dir / "femur_left.nii.gz"
    femur_right_path = femur_dir / "femur_right.nii.gz"

    if not tibia_path.exists():
        return None, f"missing tibia: {source_case}"

    if not femur_left_path.exists() and not femur_right_path.exists():
        return None, f"missing femur: {source_case}"

    ct_img = load_nifti(ct_path)
    ct_data = ct_img.get_fdata().astype(np.float32)

    femur = np.zeros(ct_data.shape, dtype=bool)

    if femur_left_path.exists():
        femur_left = load_nifti(femur_left_path).get_fdata() > 0
        if femur_left.shape != ct_data.shape:
            return None, f"femur_left shape mismatch: {source_case}"
        femur |= femur_left

    if femur_right_path.exists():
        femur_right = load_nifti(femur_right_path).get_fdata() > 0
        if femur_right.shape != ct_data.shape:
            return None, f"femur_right shape mismatch: {source_case}"
        femur |= femur_right

    tibia = load_nifti(tibia_path).get_fdata() > 0


    if tibia.shape != ct_data.shape:
        return None, f"tibia shape mismatch: {source_case}"
    
    
    tibia = keep_largest_nonoverlapping_components(
    tibia,
    femur,
    max_components=2,
    max_overlap_ratio=0.05,
)

    femur_voxels = mask_voxels(femur)
    tibia_voxels = mask_voxels(tibia)

    if femur_voxels == 0:
        return None, f"empty femur: {source_case}"

    if tibia_voxels == 0:
        return None, f"empty tibia: {source_case}"

    proximal_tibia, selected_side = make_proximal_tibia_near_femur(
        tibia,
        femur,
        keep_ratio=KEEP_RATIO,
    )

    proximal_voxels = mask_voxels(proximal_tibia)

    if proximal_voxels == 0:
        return None, f"empty proximal tibia: {source_case}"

    label = np.zeros(ct_data.shape, dtype=np.uint8)

    # Femur = 1
    label[femur > 0] = 1

    # Proximal tibia = 2, but never overwrite femur
    proximal_tibia_clean = (proximal_tibia > 0) & ~(femur > 0)
    label[proximal_tibia_clean] = 2

    ct_data, label, cropped_affine, crop_box = crop_to_mask_bbox(
    ct_data,
    label,
    ct_img.affine,
    margin=64,
    )

    print(f"[CROP] {source_case} crop_box={crop_box}, new_shape={ct_data.shape}")

    case_id = f"vsd_{output_index:04d}"

    out_img = IMAGES_TR / f"{case_id}_0000.nii.gz"
    out_lbl = LABELS_TR / f"{case_id}.nii.gz"

    ct_out = nib.Nifti1Image(ct_data, cropped_affine)
    ct_out.header.set_data_dtype(np.float32)
    ct_out.set_qform(ct_img.affine, code=1)
    ct_out.set_sform(ct_img.affine, code=1)

    label_out = nib.Nifti1Image(label, cropped_affine)
    label_out.header.set_data_dtype(np.uint8)
    label_out.set_qform(ct_img.affine, code=1)
    label_out.set_sform(ct_img.affine, code=1)

    nib.save(ct_out, str(out_img))
    nib.save(label_out, str(out_lbl))

    row = {
    "case_id": case_id,
    "source_case": source_case,
    "ct_path": str(ct_path),
    "femur_voxels": femur_voxels,
    "tibia_voxels": tibia_voxels,
    "proximal_tibia_voxels": proximal_voxels,
    "proximal_side_selected": selected_side,
    "cropped_shape": str(ct_data.shape),
    "crop_box": str(crop_box),
    "unique_labels": str(np.unique(label).tolist()),
}

    return row, None


def main():
    ct_files = sorted(VSD_NIFTI_DIR.glob("*_0000.nii.gz"))

    rows = []
    saved = 0
    scanned = 0

    for ct_path in ct_files:
        if saved >= MAX_CASES:
            break

        scanned += 1
        result, error = build_case(ct_path, saved + 1)

        if error:
            print(f"[SKIP] {error}")
            continue

        rows.append(result)
        saved += 1
        print(
            f"[OK] {result['source_case']} -> {result['case_id']} | "
            f"femur={result['femur_voxels']} | "
            f"tibia={result['tibia_voxels']} | "
            f"prox_tibia={result['proximal_tibia_voxels']} | "
            f"side={result['proximal_side_selected']}"
        )

        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "case_id",
                "source_case",
                "ct_path",
                "femur_voxels",
                "tibia_voxels",
                "proximal_tibia_voxels",
                "proximal_side_selected",
                "cropped_shape",
                "crop_box",
                "unique_labels",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nScanned: {scanned}")
    print(f"Saved: {saved}")
    print(f"Log saved to: {LOG_CSV}")


if __name__ == "__main__":
    main() 