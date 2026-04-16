from pathlib import Path
import numpy as np
import nibabel as nib
import csv

GT_DIR = Path(r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\test_gt")
PRED_DIR = Path(r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\test_pred")
OUT_CSV = Path(r"C:\vscode_windows\thesis\3d-segmentation-femur-tibia\nnunet\data\femur_metrics.csv")


def load_mask(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    return (data > 0).astype(np.uint8)


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    intersection = np.sum(gt * pred)
    denom = np.sum(gt) + np.sum(pred)
    if denom == 0:
        return 1.0
    return (2.0 * intersection) / denom


def iou_score(gt: np.ndarray, pred: np.ndarray) -> float:
    intersection = np.sum(gt * pred)
    union = np.sum((gt + pred) > 0)
    if union == 0:
        return 1.0
    return intersection / union


def precision_score(gt: np.ndarray, pred: np.ndarray) -> float:
    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)


def recall_score(gt: np.ndarray, pred: np.ndarray) -> float:
    tp = np.sum((gt == 1) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)


def volume_difference(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_vol = np.sum(gt)
    pred_vol = np.sum(pred)
    if gt_vol == 0:
        return 0.0 if pred_vol == 0 else float("inf")
    return abs(pred_vol - gt_vol) / gt_vol


def main():
    pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
    rows = []

    for pred_path in pred_files:
        case_id = pred_path.name
        gt_path = GT_DIR / case_id

        if not gt_path.exists():
            print(f"[SKIP] Missing GT for {case_id}")
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            print(f"[SKIP] Shape mismatch for {case_id}: GT {gt.shape} vs Pred {pred.shape}")
            continue

        row = {
            "case": case_id,
            "dice": dice_score(gt, pred),
            "iou": iou_score(gt, pred),
            "precision": precision_score(gt, pred),
            "recall": recall_score(gt, pred),
            "relative_volume_difference": volume_difference(gt, pred),
        }
        rows.append(row)
        print(
            f"{case_id} | Dice={row['dice']:.4f} | IoU={row['iou']:.4f} | "
            f"Precision={row['precision']:.4f} | Recall={row['recall']:.4f}"
        )

    if not rows:
        print("No valid cases found.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "dice",
                "iou",
                "precision",
                "recall",
                "relative_volume_difference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # summary
    for key in ["dice", "iou", "precision", "recall", "relative_volume_difference"]:
        values = np.array([r[key] for r in rows], dtype=np.float64)
        print(f"{key}: mean={values.mean():.4f}, std={values.std():.4f}")

    print(f"\nSaved metrics to: {OUT_CSV}")


if __name__ == "__main__":
    main()