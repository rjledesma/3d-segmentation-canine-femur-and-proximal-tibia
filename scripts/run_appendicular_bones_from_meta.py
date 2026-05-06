from pathlib import Path
import subprocess
import pandas as pd

TOTALSEG_ROOT = Path(r"C:\Users\rjled\Downloads\TotalSegmentator_dataset")
META_CSV = TOTALSEG_ROOT / "meta.csv"

MAX_CASES_TO_RUN = 20

KEYWORDS = "leg|lower|knee|tibia"


def main():
    df = pd.read_csv(META_CSV, sep=";")

    candidates = df[
        df["study_type"].str.contains(KEYWORDS, case=False, na=False)
    ].copy()

    print(f"Found {len(candidates)} likely lower-extremity cases from meta.csv")

    processed = 0

    for _, row in candidates.iterrows():
        case_id = row["image_id"]
        study_type = row["study_type"]

        case_dir = TOTALSEG_ROOT / case_id
        ct_path = case_dir / "ct.nii.gz"
        out_dir = case_dir / "appendicular_bones"
        tibia_path = out_dir / "tibia.nii.gz"

        if not ct_path.exists():
            print(f"[SKIP] Missing CT: {case_id}")
            continue

        if tibia_path.exists():
            print(f"[SKIP] Already has tibia output: {case_id}")
            continue

        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(out_dir),
            "--task", "appendicular_bones",
        ]

        print(f"\n[RUN] {case_id} | {study_type}")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            print(f"[FAILED] {case_id}: {e}")
            continue

        if processed >= MAX_CASES_TO_RUN:
            break

    print(f"\nProcessed {processed} new cases.")


if __name__ == "__main__":
    main()