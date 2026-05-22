"""
Download and extract the CUAD dataset to data/cuad_repo/.

Expected output layout (matches data/cuad.py expectations):
    <repo_root>/data/cuad_repo/train_separate_questions.json
    <repo_root>/data/cuad_repo/test.json

Usage (from repo root or autoresearch/):
    python autoresearch/data/download_cuad.py
    python autoresearch/data/download_cuad.py --dest /custom/path/cuad_repo
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://github.com/TheAtticusProject/cuad/raw/refs/heads/main/data.zip"
REQUIRED_FILES = ["train_separate_questions.json", "test.json"]


def main():
    p = argparse.ArgumentParser(description="Download CUAD dataset.")
    p.add_argument(
        "--dest",
        default=None,
        help="Destination directory (default: data/cuad_repo/ at repo root)",
    )
    args = p.parse_args()

    # autoresearch/data/download_cuad.py → .parent.parent.parent = repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    dest = Path(args.dest) if args.dest else repo_root / "data" / "cuad_repo"
    dest.mkdir(parents=True, exist_ok=True)

    already = [f for f in REQUIRED_FILES if (dest / f).exists()]
    if len(already) == len(REQUIRED_FILES):
        print(f"[ok] All required files already present in {dest}")
        return 0

    print(f"Downloading {URL} ...")
    with urlopen(URL) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data) / 1_048_576:.1f} MB")

    print("Extracting ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
        for required in REQUIRED_FILES:
            candidates = [m for m in members if m.endswith(required)]
            if not candidates:
                print(f"  [error] {required!r} not found in zip. Members: {members[:10]}")
                return 1
            src = candidates[0]
            target = dest / required
            target.write_bytes(zf.read(src))
            try:
                obj = json.loads(target.read_bytes())
                n = len(obj) if isinstance(obj, (list, dict)) else "?"
                print(f"  {target}  ({n} records)")
            except json.JSONDecodeError as e:
                print(f"  [error] {target} is not valid JSON: {e}")
                return 1

    print(f"\n[done] CUAD data ready at {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
