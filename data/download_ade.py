"""
Download ADE (Adverse Drug Events) dataset in SpERT format.

Source: SpERT data server (lavis.cs.hs-rm.de).
ADE uses 10-fold cross-validation (no fixed train/dev/test).
We use fold 0: train split → further split 90/10 into train/dev.

Schema:
    Entity types (2): Adverse-Effect, Drug
    Relation types (1): Adverse-Effect (drug causes adverse effect)
"""
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).parent / "ade"

SPERT_BASE = "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade"
FILES = {
    "ade_split_0_train.json": f"{SPERT_BASE}/ade_split_0_train.json",
    "ade_split_0_test.json": f"{SPERT_BASE}/ade_split_0_test.json",
    "ade_types.json": f"{SPERT_BASE}/ade_types.json",
}


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        dst = DATA_DIR / filename
        if dst.exists():
            print(f"   {filename} already exists, skipping")
            continue
        print(f"Downloading {url}")
        urlretrieve(url, dst)
        print(f"   saved to {dst} ({dst.stat().st_size // 1024} KB)")

    # Split train into train/dev (90/10)
    train_path = DATA_DIR / "ade_split_0_train.json"
    dev_path = DATA_DIR / "ade_dev.json"
    final_train_path = DATA_DIR / "ade_train.json"
    test_path = DATA_DIR / "ade_test.json"

    if not dev_path.exists():
        with open(train_path) as f:
            data = json.load(f)
        random.seed(42)
        random.shuffle(data)
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        dev_data = data[split_idx:]

        with open(final_train_path, "w") as f:
            json.dump(train_data, f)
        with open(dev_path, "w") as f:
            json.dump(dev_data, f)
        print(f"   split train: {len(train_data)} train + {len(dev_data)} dev")

    # Copy test as-is
    src_test = DATA_DIR / "ade_split_0_test.json"
    if not test_path.exists() and src_test.exists():
        import shutil
        shutil.copy2(src_test, test_path)

    # Verify
    for name in ["ade_train.json", "ade_dev.json", "ade_test.json"]:
        p = DATA_DIR / name
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            print(f"   {name}: {len(data)} sentences")

    print(f"ADE ready at {DATA_DIR}")


if __name__ == "__main__":
    download()
