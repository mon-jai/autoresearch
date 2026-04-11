"""
Download SciERC dataset (Luan et al., 2018) to data/scierc/.

Source: https://huggingface.co/datasets/sthoran/scierc_processed_data
        (mirror of the original Stanford SciERC processed_data, since the
         Stanford NLP project page returns 404 as of 2026-04)

SciERC schema (per JSON line, one document):
    {
        "doc_key": "...",
        "sentences": [["w1", "w2", ...], [...], ...],     # tokenized
        "ner":       [[[s, e, "TYPE"], ...], [...], ...], # per sentence, end inclusive
        "relations": [[[hs, he, ts, te, "REL"], ...], ...] # per sentence
        "clusters":  [...]                                 # coreference, unused for Stage 2a
    }

Entity types (6): Task, Method, Metric, Material, OtherScientificTerm, Generic
Relation types (7): USED-FOR, FEATURE-OF, HYPONYM-OF, EVALUATE-FOR,
                    PART-OF, COMPARE, CONJUNCTION

Splits (verified 2026-04-11):
    train.json: 350 documents
    dev.json:    50 documents
    test.json:  100 documents
    Total:      500 documents
"""
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).parent / "scierc"
HF_BASE = "https://huggingface.co/datasets/sthoran/scierc_processed_data/resolve/main"
SPLITS = ["train.json", "dev.json", "test.json"]


def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all((DATA_DIR / s).exists() for s in SPLITS):
        print(f"✅ SciERC already present at {DATA_DIR}")
        for split in SPLITS:
            with open(DATA_DIR / split) as fp:
                n = sum(1 for _ in fp)
            print(f"   {split}: {n} documents")
        return

    for split in SPLITS:
        dst = DATA_DIR / split
        if dst.exists():
            continue
        url = f"{HF_BASE}/{split}"
        print(f"⬇️  {url}")
        urlretrieve(url, dst)
        print(f"   saved to {dst} ({dst.stat().st_size // 1024} KB)")

    # Verify schema by parsing first record of each split
    for split in SPLITS:
        f = DATA_DIR / split
        if not f.exists():
            print(f"❌ Missing {f}", file=sys.stderr)
            sys.exit(1)
        with open(f) as fp:
            n = sum(1 for _ in fp)
        with open(f) as fp:
            first = json.loads(fp.readline())
        required_keys = {"sentences", "ner", "relations"}
        if not required_keys.issubset(first.keys()):
            print(f"❌ {split} missing keys; got {list(first.keys())}", file=sys.stderr)
            sys.exit(1)
        print(f"   {split}: {n} documents (keys OK)")

    print(f"✅ SciERC ready at {DATA_DIR}")


if __name__ == "__main__":
    download_and_extract()
