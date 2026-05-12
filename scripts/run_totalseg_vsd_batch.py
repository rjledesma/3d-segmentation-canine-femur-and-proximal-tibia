from pathlib import Path
import subprocess

VSD_NIFTI_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_nifti"
)

VSD_TOTALSEG_DIR = Path(
    r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\vsd_totalseg"
)

# Start small first. Increase later.
MAX_CASES = 50

VSD_TOTALSEG_DIR.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str]) -> bool:
    print("\n[RUN]")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {e}")
        return False


def case_name_from_nifti(path: Path) -> str:
    """
    Converts:
    vsd_001_0001_0000.nii.gz
    into:
    vsd_001_0001
    """
    name = path.name

    if name.endswith("_0000.nii.gz"):
        return name.replace("_0000.nii.gz", "")

    if name.endswith("_0000.nii"):
        return name.replace("_0000.nii", "")

    raise ValueError(f"Unexpected nnU-Net filename: {path.name}")


def process_case(ct_path: Path) -> bool:
    case_name = case_name_from_nifti(ct_path)

    appendicular_out = VSD_TOTALSEG_DIR / f"{case_name}_appendicular"
    femur_out = VSD_TOTALSEG_DIR / f"{case_name}_femur"

    tibia_path = appendicular_out / "tibia.nii.gz"
    femur_left_path = femur_out / "femur_left.nii.gz"
    femur_right_path = femur_out / "femur_right.nii.gz"

    print(f"\n========== {case_name} ==========")

    # 1. Appendicular bones for tibia
    if tibia_path.exists():
        print(f"[SKIP] Appendicular already exists: {tibia_path}")
    else:
        cmd_appendicular = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(appendicular_out),
            "--task", "appendicular_bones",
            "--nr_thr_resamp", "1",
            "--nr_thr_saving", "1",
        ]

        ok = run_command(cmd_appendicular)
        if not ok:
            print(f"[SKIP] Appendicular failed for {case_name}")
            return False

    # 2. Femur ROI subset
    if femur_left_path.exists() or femur_right_path.exists():
        print(f"[SKIP] Femur already exists: {femur_out}")
    else:
        cmd_femur = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(femur_out),
            "--roi_subset", "femur_left", "femur_right",
            "--nr_thr_resamp", "1",
            "--nr_thr_saving", "1",
        ]

        ok = run_command(cmd_femur)
        if not ok:
            print(f"[SKIP] Femur failed for {case_name}")
            return False

    print(f"[OK] Finished {case_name}")
    return True


def main():
    ct_files = sorted(VSD_NIFTI_DIR.glob("*_0000.nii.gz"))

    print(f"Found {len(ct_files)} VSD NIfTI files.")

    processed = 0

    for ct_path in ct_files:
        if processed >= MAX_CASES:
            break

        success = process_case(ct_path)

        if success:
            processed += 1

    print(f"\nDone. Processed {processed} cases.")


if __name__ == "__main__":
    main()