"""
Download the Twitter Customer Support dataset from Kaggle using kagglehub.

Usage:
    python -m src.download_data
"""
import os
import shutil
import kagglehub
from src.config import DATA_RAW_DIR, RAW_CSV


def download_dataset():
    """Download dataset and copy to data/raw/."""
    if os.path.exists(RAW_CSV):
        print(f"[INFO] Dataset already exists at {RAW_CSV}")
        return RAW_CSV

    print("[INFO] Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    print(f"[INFO] Downloaded to: {path}")

    # Find all CSV files in the downloaded directory
    csv_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                full_path = os.path.join(root, f)
                csv_files.append((full_path, f, os.path.getsize(full_path)))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")

    # Prefer twcs.csv specifically; otherwise pick the largest CSV
    target = None
    for fpath, fname, fsize in csv_files:
        if fname == "twcs.csv":
            target = fpath
            break
    if target is None:
        target = max(csv_files, key=lambda x: x[2])[0]

    dst = os.path.join(DATA_RAW_DIR, "twcs.csv")
    shutil.copy2(target, dst)
    print(f"[INFO] Copied {os.path.basename(target)} ({os.path.getsize(dst) / 1e6:.1f} MB) -> {dst}")
    return dst


if __name__ == "__main__":
    download_dataset()
