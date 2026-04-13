"""
Download CoNLL04 dataset (Roth & Yih, 2004) in SpERT format to data/conll04/.

Source: SpERT repository processed data (Eberts & Ulges, 2020).
The SpERT repo hosts CoNLL04 in a unified JSON format used by most
recent joint NER+RE papers.

CoNLL04 schema (JSON array, each element is one sentence):
    {
        "tokens":    ["w1", "w2", ...],
        "entities":  [{"type": "Loc", "start": 0, "end": 2}, ...],  # end exclusive
        "relations": [{"type": "Located_In", "head": 0, "tail": 1}, ...]
                     # head/tail index into entities list
    }

Entity types (4): Loc, Org, Peop, Other
Relation types (5): Located_In, Work_For, OrgBased_In, Live_In, Kill

Splits:
    conll04_train.json
    conll04_dev.json
    conll04_test.json

Source: SpERT data server (lavis.cs.hs-rm.de). The GitHub raw URLs
return 404; the actual data is hosted separately.
"""
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).parent / "conll04"

# SpERT repo processed data (standard distribution)
SPERT_BASE = "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04"
SPLITS = {
    "conll04_train.json": f"{SPERT_BASE}/conll04_train.json",
    "conll04_dev.json":   f"{SPERT_BASE}/conll04_dev.json",
    "conll04_test.json":  f"{SPERT_BASE}/conll04_test.json",
}


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all((DATA_DIR / s).exists() for s in SPLITS):
        print(f"CoNLL04 already present at {DATA_DIR}")
        for split in SPLITS:
            with open(DATA_DIR / split) as fp:
                data = json.load(fp)
            print(f"   {split}: {len(data)} sentences")
        return

    for filename, url in SPLITS.items():
        dst = DATA_DIR / filename
        if dst.exists():
            continue
        print(f"Downloading {url}")
        urlretrieve(url, dst)
        print(f"   saved to {dst} ({dst.stat().st_size // 1024} KB)")

    # Verify
    for filename in SPLITS:
        f = DATA_DIR / filename
        if not f.exists():
            print(f"Missing {f}", file=sys.stderr)
            sys.exit(1)
        with open(f) as fp:
            data = json.load(fp)
        if not isinstance(data, list) or len(data) == 0:
            print(f"{filename} is not a non-empty JSON array", file=sys.stderr)
            sys.exit(1)
        first = data[0]
        required_keys = {"tokens", "entities", "relations"}
        if not required_keys.issubset(first.keys()):
            print(f"{filename} missing keys; got {list(first.keys())}", file=sys.stderr)
            sys.exit(1)
        print(f"   {filename}: {len(data)} sentences (keys OK)")

    print(f"CoNLL04 ready at {DATA_DIR}")


if __name__ == "__main__":
    download()
