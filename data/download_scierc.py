"""
Download SciERC dataset (Luan et al., 2018) to data/scierc/.

SciERC schema (per JSON line, one document):
    {
        "doc_key": "...",
        "sentences": [["w1", "w2", ...], [...], ...],     # tokenized
        "ner":       [[[s, e, "TYPE"], ...], [...], ...], # per sentence, end inclusive
        "relations": [[[hs, he, ts, te, "REL"], ...], ...] # per sentence
    }

Entity types (6): Task, Method, Metric, Material, OtherScientificTerm, Generic
Relation types (7): USED-FOR, FEATURE-OF, HYPONYM-OF, EVALUATE-FOR,
                    PART-OF, COMPARE, CONJUNCTION

Total: ~500 abstracts; ~8k entities; ~4.7k relations.
"""
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).parent / "scierc"
URL = "http://nlp.stanford.edu/projects/scierc/processed_data/json/sciERC_processed.tar.gz"
# Fallback if Stanford URL changes — from the official mirror
FALLBACK_URL = "https://nlp.stanford.edu/projects/scierc/processed_data/json/sciERC_processed.tar.gz"


def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "sciERC_processed.tar.gz"

    if (DATA_DIR / "train.json").exists() and (DATA_DIR / "dev.json").exists():
        print(f"✅ SciERC already extracted at {DATA_DIR}")
        return

    if not archive.exists():
        print(f"⬇️  Downloading SciERC from {URL} ...")
        try:
            urlretrieve(URL, archive)
        except Exception as e:
            print(f"   primary URL failed ({e}), trying fallback ...")
            urlretrieve(FALLBACK_URL, archive)
        print(f"   saved to {archive} ({archive.stat().st_size // 1024} KB)")

    print("📦 Extracting ...")
    import tarfile
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    # The tarball typically extracts to processed_data/json/{train,dev,test}.json
    # Move them up to DATA_DIR/ if so
    nested = DATA_DIR / "processed_data" / "json"
    if nested.exists():
        for split in ["train.json", "dev.json", "test.json"]:
            src = nested / split
            dst = DATA_DIR / split
            if src.exists() and not dst.exists():
                src.rename(dst)
                print(f"   moved {src} -> {dst}")

    # Verify
    for split in ["train.json", "dev.json", "test.json"]:
        f = DATA_DIR / split
        if not f.exists():
            print(f"❌ Missing {f}", file=sys.stderr)
            sys.exit(1)
        with open(f) as fp:
            n = sum(1 for _ in fp)
        print(f"   {split}: {n} documents")

    print(f"✅ SciERC ready at {DATA_DIR}")


if __name__ == "__main__":
    download_and_extract()
