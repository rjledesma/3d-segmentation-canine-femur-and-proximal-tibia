import json
from configs.paths import CANINE_DATASET_DIR

def main() -> None:
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "femur": 1,
            "proximal_tibia": 2
        },
        "numTraining": 0,
        "file_ending": ".nii.gz"
    }

    out_path = CANINE_DATASET_DIR / "dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()