from pathlib import Path
import nibabel as nib
import numpy as np

TOTALSEG_ROOT = Path(r"C:\Users\rjled\Downloads\TotalSegmentator_dataset")

def main():
    case_dirs = sorted(
        [p for p in TOTALSEG_ROOT.iterdir() if p.is_dir() and p.name.startswith("s")]
    )

    found = 0
    checked = 0

    for case_dir in case_dirs:
        tibia_path = case_dir / "appendicular_bones" / "tibia.nii.gz"

        if not tibia_path.exists():
            continue

        checked += 1

        try:
            img = nib.load(str(tibia_path))
            data = img.get_fdata()
            voxels = int(np.sum(data > 0))
        except Exception as e:
            print(f"[BAD] {case_dir.name}: {e}")
            continue

        if voxels > 0:
            found += 1
            print(f"[FOUND] {case_dir.name} | tibia voxels: {voxels}")
        else:
            print(f"[EMPTY] {case_dir.name}")

    print(f"\nChecked appendicular_bones folders: {checked}")
    print(f"Cases with non-empty tibia: {found}")

if __name__ == "__main__":
    main()