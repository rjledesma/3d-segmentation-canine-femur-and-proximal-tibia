from pathlib import Path
import subprocess

TOTALSEG_ROOT = Path(r"C:\Users\rjled\Downloads\TotalSegmentator_dataset")

# Start small first
MAX_CASES_TO_RUN = 10

def main():
    case_dirs = sorted(
        [p for p in TOTALSEG_ROOT.iterdir() if p.is_dir() and p.name.startswith("s")]
    )

    processed = 0

    for case_dir in case_dirs:
        ct_path = case_dir / "ct.nii.gz"
        out_dir = case_dir / "appendicular_bones"

        if not ct_path.exists():
            print(f"[SKIP] Missing CT: {case_dir.name}")
            continue

        # Skip if already exists
        if (out_dir / "tibia.nii.gz").exists():
            print(f"[SKIP] Already has tibia output: {case_dir.name}")
            continue

        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(out_dir),
            "--task", "appendicular_bones",
        ]

        print(f"\n[RUN] {case_dir.name}")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            print(f"[FAILED] {case_dir.name}: {e}")
            continue

        if processed >= MAX_CASES_TO_RUN:
            break

    print(f"\nProcessed {processed} new cases.")

if __name__ == "__main__":
    main()