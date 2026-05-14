from pathlib import Path
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh

LABEL_PATH = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\nnUNet_results\Dataset002_HumanFemurProximalTibia\nnUNetTrainer_300epochs__nnUNetPlans__3d_fullres\fold_0\validation\vsd_0001.nii.gz"
)

OUT_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\stl_preview"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_label_as_stl(data: np.ndarray, affine: np.ndarray, label_value: int, out_path: Path):
    mask = data == label_value

    if np.sum(mask) == 0:
        print(f"[SKIP] label {label_value} is empty")
        return

    print(f"[INFO] Creating STL for label {label_value}, voxels={int(np.sum(mask))}")

    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        step_size=2  # larger = faster/smaller STL, lower detail
    )

    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_world = (affine @ verts_h.T).T[:, :3]

    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    mesh.export(str(out_path))

    print(f"[OK] saved: {out_path}")


def main():
    img = nib.load(str(LABEL_PATH))
    data = img.get_fdata().astype(np.uint8)

    print("Unique labels:", np.unique(data, return_counts=True))

    save_label_as_stl(data, img.affine, 1, OUT_DIR / "vsd_0001_femur.stl")
    save_label_as_stl(data, img.affine, 2, OUT_DIR / "vsd_0001_proximal_tibia.stl")


if __name__ == "__main__":
    main()