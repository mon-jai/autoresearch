"""
Download a held-out scientific text corpus for the Stage 2b realism critic.

Source: ccdv/arxiv-classification on HuggingFace Datasets
   https://huggingface.co/datasets/ccdv/arxiv-classification

Important: each "text" field in the source dataset is a FULL paper (~60 KB).
We don't need full papers — we need short scientific text segments that look
like SciERC abstracts. So we truncate each text to TRUNCATE_CHARS (default
1024) and save to a small jsonl file.

Why this corpus is appropriate as the critic's "real" pool:
- It's ENGINEERED to be different from SciERC (different paper, different
  authors, different sections of the source paper) → critic won't trivially
  memorize "SciERC vs not-SciERC"
- The text is real arXiv papers in cs.* categories → topic-aligned with the
  scientific content Stage 2 cares about
- The validation split (2500 papers) is small enough to download fast
  (~144 MB parquet) but large enough that the critic won't see repeats
  during a 1500-step run
"""
import io
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).parent / "arxiv_real"
PARQUET_URL = (
    "https://huggingface.co/datasets/ccdv/arxiv-classification/"
    "resolve/main/data/validation-00000-of-00001.parquet"
)
PARQUET_PATH = DATA_DIR / "validation.parquet"
JSONL_PATH = DATA_DIR / "cs_validation.jsonl"

# CS-only labels in ccdv/arxiv-classification (excluding math.*)
CS_LABELS = {1, 2, 3, 5, 6, 7, 8, 9}  # cs.CV, cs.AI, cs.SY, cs.CE, cs.PL, cs.IT, cs.DS, cs.NE
TRUNCATE_CHARS = 1024  # ~ first paragraph or two = approximately one abstract


def download_and_filter():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if JSONL_PATH.exists():
        with open(JSONL_PATH) as f:
            n = sum(1 for _ in f)
        print(f"✅ arXiv corpus already present: {JSONL_PATH} ({n} documents)")
        return

    if not PARQUET_PATH.exists():
        print(f"⬇️  Downloading arXiv validation parquet from HF ({PARQUET_URL})")
        urlretrieve(PARQUET_URL, PARQUET_PATH)
        size_mb = PARQUET_PATH.stat().st_size // (1024 * 1024)
        print(f"   saved {PARQUET_PATH} ({size_mb} MB)")

    print("📦 Reading parquet + filtering to cs.* + truncating ...")
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET_PATH)
    texts = table.column("text").to_pylist()
    labels = table.column("label").to_pylist()

    n_kept = 0
    with open(JSONL_PATH, "w") as out:
        for text, label in zip(texts, labels):
            if label not in CS_LABELS:
                continue
            short = text[:TRUNCATE_CHARS].strip()
            if not short:
                continue
            out.write(json.dumps({"text": short, "label": label}) + "\n")
            n_kept += 1

    # Drop the big parquet — we don't need it anymore
    PARQUET_PATH.unlink()

    size_kb = JSONL_PATH.stat().st_size // 1024
    print(f"✅ Saved {n_kept} cs.* documents to {JSONL_PATH} ({size_kb} KB)")


if __name__ == "__main__":
    download_and_filter()
