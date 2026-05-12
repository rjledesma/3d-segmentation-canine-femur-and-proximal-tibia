from pathlib import Path
import nibabel as nib
import numpy as np

CT_PATH = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti\vsd_001_0001_0000.nii.gz"
)

FEMUR_LEFT_PATH = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg\vsd_001_0001_femur\femur_left.nii.gz"
)

FEMUR_RIGHT_PATH = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg\vsd_001_0001_femur\femur_right.nii.gz"
)

TIBIA_MASK_PATH = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg\vsd_001_0001_appendicular\tibia.nii.gz"
)

DATASET002_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_raw\Dataset002_HumanFemurProximalTibia"
)

IMAGES_TR = DATASET002_DIR / "imagesTr"
LABELS_TR = DATASET002_DIR / "labelsTr"

CASE_ID = "vsd_0001"

KEEP_RATIO = 0.30


def load(path: Path):
    return nib.load(str(path))


def make_proximal_tibia_near_femur(
    tibia_mask: np.ndarray,
    femur_mask: np.ndarray,
    keep_ratio: float = 0.30
) -> np.ndarray:
    """
    Keeps the end of the tibia that is closest to the femur.
    This is safer than assuming superior/inferior direction from Z axis.
    """

    tibia_coords = np.argwhere(tibia_mask > 0)
    femur_coords = np.argwhere(femur_mask > 0)

    if tibia_coords.size == 0:
        return np.zeros_like(tibia_mask, dtype=np.uint8)

    if femur_coords.size == 0:
        raise ValueError("Femur mask is empty, cannot determine proximal tibia side.")

    # Use Z only for slicing, but decide which end using distance to femur
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

    femur_center = femur_coords.mean(axis=0)

    lower_center = lower_coords.mean(axis=0)
    upper_center = upper_coords.mean(axis=0)

    lower_dist = np.linalg.norm(lower_center - femur_center)
    upper_dist = np.linalg.norm(upper_center - femur_center)

    if lower_dist < upper_dist:
        proximal = lower_end
        print("[INFO] Proximal tibia selected: lower-Z end, closest to femur")
    else:
        proximal = upper_end
        print("[INFO] Proximal tibia selected: upper-Z end, closest to femur")

    return proximal.astype(np.uint8)


def main():
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)

    ct_img = load(CT_PATH)
    ct_data = ct_img.get_fdata().astype(np.float32)

    femur = np.zeros(ct_data.shape, dtype=bool)

    if FEMUR_LEFT_PATH.exists():
        femur_left = load(FEMUR_LEFT_PATH).get_fdata() > 0
        femur |= femur_left

    if FEMUR_RIGHT_PATH.exists():
        femur_right = load(FEMUR_RIGHT_PATH).get_fdata() > 0
        femur |= femur_right

    tibia_img = load(TIBIA_MASK_PATH)
    tibia = tibia_img.get_fdata() > 0

    if femur.shape != ct_data.shape:
        raise ValueError(f"Femur shape mismatch: femur={femur.shape}, CT={ct_data.shape}")

    if tibia.shape != ct_data.shape:
        raise ValueError(f"Tibia shape mismatch: tibia={tibia.shape}, CT={ct_data.shape}")

    proximal_tibia = make_proximal_tibia_near_femur(
    tibia,
    femur,
    keep_ratio=KEEP_RATIO
)

    label = np.zeros(ct_data.shape, dtype=np.uint8)
    label[femur] = 1
    label[proximal_tibia > 0] = 2

    print("CT shape:", ct_data.shape)
    print("Femur voxels:", int(np.sum(femur)))
    print("Full tibia voxels:", int(np.sum(tibia)))
    print("Proximal tibia voxels:", int(np.sum(proximal_tibia)))
    print("Final label unique values:", np.unique(label))

    out_img = IMAGES_TR / f"{CASE_ID}_0000.nii.gz"
    out_lbl = LABELS_TR / f"{CASE_ID}.nii.gz"

    ct_out = nib.Nifti1Image(ct_data, ct_img.affine)
    ct_out.header.set_data_dtype(np.float32)
    ct_out.set_qform(ct_img.affine, code=1)
    ct_out.set_sform(ct_img.affine, code=1)

    label_out = nib.Nifti1Image(label, ct_img.affine)
    label_out.header.set_data_dtype(np.uint8)
    label_out.set_qform(ct_img.affine, code=1)
    label_out.set_sform(ct_img.affine, code=1)

    nib.save(ct_out, str(out_img))
    nib.save(label_out, str(out_lbl))

    print("Saved image:", out_img)
    print("Saved label:", out_lbl)


if __name__ == "__main__":
    main()