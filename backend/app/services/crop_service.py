from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def _compute_new_affine(affine: np.ndarray, starts: tuple[int, int, int]) -> np.ndarray:
    """
    Updates affine after cropping so the cropped NIfTI still has correct world position.
    """
    new_affine = affine.copy()
    offset = np.array([starts[0], starts[1], starts[2], 1])
    new_origin = affine @ offset
    new_affine[:3, 3] = new_origin[:3]
    return new_affine


def crop_ct_for_femur_tibia_inference(
    input_nifti_path: Path,
    output_nifti_path: Path | None = None,
    bone_threshold: int = 250,
    margin_xy: int = 48,
    margin_long: int = 32,
    max_long_axis_voxels: int = 900,
    min_voxels: int = 5000,
) -> Path:
    """
    Crops a large lower-limb CT to a smaller bone-focused ROI before nnU-Net inference.

    Intended for DICOM-converted full lower-limb CTs.

    Heuristic:
    1. Threshold bone using HU > bone_threshold.
    2. Keep the largest connected body/bone region.
    3. Find the longest anatomical axis.
    4. Keep the side with the largest bone cross-section first, usually pelvis/femur side.
    5. Limit the long axis length to reduce sliding-window inference time.

    Output overwrites input_nifti_path unless output_nifti_path is given.
    """
    input_nifti_path = Path(input_nifti_path)

    if output_nifti_path is None:
        output_nifti_path = input_nifti_path
    else:
        output_nifti_path = Path(output_nifti_path)

    img = nib.load(str(input_nifti_path))
    data = img.get_fdata().astype(np.float32)

    print(f"[CROP] Original shape: {data.shape}")

    bone_mask = data > bone_threshold

    if int(np.sum(bone_mask)) < min_voxels:
        print("[CROP] Not enough bone voxels found. Skipping crop.")
        return input_nifti_path

    labeled, num = ndimage.label(bone_mask)

    if num == 0:
        print("[CROP] No connected bone components found. Skipping crop.")
        return input_nifti_path

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0

    # Keep the largest connected component to avoid random noise.
    largest_label = int(np.argmax(component_sizes))
    clean_mask = labeled == largest_label

    coords = np.argwhere(clean_mask)

    if coords.size == 0:
        print("[CROP] Empty cleaned mask. Skipping crop.")
        return input_nifti_path

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    bbox_lengths = maxs - mins
    long_axis = int(np.argmax(bbox_lengths))

    starts = mins.copy()
    ends = maxs.copy()

    # Add margins on non-long axes.
    for axis in range(3):
        if axis == long_axis:
            continue

        starts[axis] = max(0, starts[axis] - margin_xy)
        ends[axis] = min(data.shape[axis], ends[axis] + margin_xy)

    # Decide which side of the long axis is the hip/femur side.
    # Usually the pelvis/femur side has larger bone area per slice than feet/ankle side.
    long_min = int(mins[long_axis])
    long_max = int(maxs[long_axis])
    long_length = long_max - long_min

    axes_to_sum = tuple(axis for axis in range(3) if axis != long_axis)
    bone_profile = clean_mask.sum(axis=axes_to_sum)

    profile_region = bone_profile[long_min:long_max]

    if profile_region.size == 0:
        print("[CROP] Empty bone profile. Skipping crop.")
        return input_nifti_path

    peak_index_local = int(np.argmax(profile_region))
    peak_index = long_min + peak_index_local

    crop_length = min(long_length + (2 * margin_long), max_long_axis_voxels)

    # If peak is closer to the start, keep from start toward the end.
    # If peak is closer to the end, keep from end backward.
    if (peak_index - long_min) <= (long_max - peak_index):
        long_start = max(0, long_min - margin_long)
        long_end = min(data.shape[long_axis], long_start + crop_length)
    else:
        long_end = min(data.shape[long_axis], long_max + margin_long)
        long_start = max(0, long_end - crop_length)

    starts[long_axis] = long_start
    ends[long_axis] = long_end

    slices = tuple(slice(int(starts[i]), int(ends[i])) for i in range(3))
    cropped = data[slices]

    print(f"[CROP] Bone bbox mins={mins.tolist()} maxs={maxs.tolist()}")
    print(f"[CROP] Long axis={long_axis}, peak index={peak_index}")
    print(f"[CROP] Crop starts={starts.tolist()} ends={ends.tolist()}")
    print(f"[CROP] Cropped shape: {cropped.shape}")

    new_affine = _compute_new_affine(img.affine, tuple(int(x) for x in starts))

    cropped_img = nib.Nifti1Image(cropped.astype(np.float32), new_affine, img.header)
    cropped_img.set_data_dtype(np.float32)

    output_nifti_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(cropped_img, str(output_nifti_path))

    print(f"[CROP] Saved cropped NIfTI: {output_nifti_path}")

    return output_nifti_path