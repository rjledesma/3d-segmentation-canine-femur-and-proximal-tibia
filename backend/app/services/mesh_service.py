from pathlib import Path
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh


def label_to_stl(
    label_path: Path,
    label_value: int,
    output_path: Path,
    step_size: int = 2,
) -> Path:
    """
    Converts one label value from a segmentation mask to STL.

    label 1 = femur
    label 2 = proximal tibia
    """
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.uint8)

    mask = data == label_value

    if np.sum(mask) == 0:
        raise ValueError(f"Label {label_value} is empty in {label_path}")

    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        step_size=step_size,
    )

    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_world = (img.affine @ verts_h.T).T[:, :3]

    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        process=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    return output_path


def convert_prediction_to_stls(prediction_path: Path, mesh_dir: Path) -> dict:
    femur_stl = mesh_dir / "femur.stl"
    tibia_stl = mesh_dir / "proximal_tibia.stl"

    femur_path = label_to_stl(
        label_path=prediction_path,
        label_value=1,
        output_path=femur_stl,
        step_size=2,
    )

    tibia_path = label_to_stl(
        label_path=prediction_path,
        label_value=2,
        output_path=tibia_stl,
        step_size=2,
    )

    return {
        "femur_stl": femur_path,
        "proximal_tibia_stl": tibia_path,
    }