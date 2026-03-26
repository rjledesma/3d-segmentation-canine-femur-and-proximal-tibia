from pathlib import Path
from configs.paths import HUMAN_DATASET_DIR, CANINE_DATASET_DIR

def check_dataset(dataset_dir: Path) -> None:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    dataset_json = dataset_dir / "dataset.json"

    print(f"\nChecking: {dataset_dir.name}")

    if not dataset_json.exists():
        print("  Missing dataset.json")
    else:
        print("  dataset.json found")

    image_files = sorted(images_tr.glob("*.nii.gz"))
    label_files = sorted(labels_tr.glob("*.nii.gz"))

    print(f"  imagesTr count: {len(image_files)}")
    print(f"  labelsTr count: {len(label_files)}")

    image_ids = {f.name.replace("_0000.nii.gz", "") for f in image_files}
    label_ids = {f.name.replace(".nii.gz", "") for f in label_files}

    missing_labels = image_ids - label_ids
    missing_images = label_ids - image_ids

    if missing_labels:
        print(f"  Missing labels for: {sorted(missing_labels)}")
    if missing_images:
        print(f"  Missing images for: {sorted(missing_images)}")

    if not missing_labels and not missing_images:
        print("  Image-label pairing looks correct")

def main() -> None:
    check_dataset(HUMAN_DATASET_DIR)
    check_dataset(CANINE_DATASET_DIR)

if __name__ == "__main__":
    main()