# run_pipeline.py — Master KG Research Pipeline

Automates the full sequence: train → infer → [verify] → build KG → evaluate
for every distinct `BertKGExtractor` experiment configuration in this project.

---

## Setup

Download the required datasets before running the pipeline. Run these from the
`autoresearch/` directory (or prefix with `autoresearch/` if running from the
repo root):

```bash
# SciERC — required for scierc_* and multi_scierc_* attempts
python data/download_scierc.py

# CoNLL04 — required for multi_conll04_* attempts
python data/download_conll04.py

# ADE — required for multi_ade_* attempts
python data/download_ade.py

# CUAD — required for span_cuad_deberta_pretrain AND span_accord_deberta_phase_b
python data/download_cuad.py

# arXiv real corpus — required only for stage2b adversarial training
python data/download_arxiv_real.py
```

ACCORD data (`data/code_accord/`) is already included in the repository and
requires no separate download.

All download scripts are idempotent — they skip the download if the target
files already exist.

---

## Quick start

```bash
cd autoresearch

# List all available attempts
uv run python run_pipeline.py --list-attempts

# Full default pipeline for Phase A+ best config (seed 42)
# Steps: train, infer, build, triple eval, KG compare
uv run python run_pipeline.py --attempt span_accord_deberta_aplus --seed 42

# Dry-run to preview all commands without executing
uv run python run_pipeline.py --attempt span_accord_deberta_aplus --seed 42 --dry-run

# Smoke-test all 12 attempt configs against ACCORD data (2-step train, verifies code paths)
uv run python run_pipeline.py --attempt all --force-dataset accord --max-steps 2 --eval-every 1

# Full pipeline including LLM verification and Graph RAG eval (requires Ollama)
uv run python run_pipeline.py --attempt span_accord_deberta_aplus --seed 42 \
    --steps train,infer,verify,build,triple,rag,compare \
    --ollama-url http://localhost:11434
```

---

## Available attempts

| Attempt | Train script | Dataset | Description |
|---|---|---|---|
| `scierc_bert` | `train_stage2` | SciERC | BERT-base BIO supervised (Stage 2 baseline) |
| `multi_scierc_scibert` | `train_multi` | SciERC | SciBERT BIO multi-dataset |
| `multi_conll04_bert` | `train_multi` | CoNLL04 | BERT-base BIO multi-dataset |
| `multi_ade_scibert` | `train_multi` | ADE | SciBERT BIO multi-dataset |
| `span_scierc_scibert` | `train_span` | SciERC | Phase 12: Span NER baseline, SciBERT |
| `span_scierc_scibert_bio` | `train_span` | SciERC | Phase 14: Span+BIO multi-task, SciBERT |
| `span_accord_bert` | `train_span` | ACCORD | Phase A early: Span NER, BERT-base |
| `span_accord_deberta` | `train_span` | ACCORD | Phase A: DeBERTa-large, no A-flags |
| `span_accord_deberta_a12` | `train_span` | ACCORD | Phase A12: + context-span RE |
| `span_accord_deberta_aplus` | `train_span` | ACCORD | **Phase A+: + A20+A21+A12 (best: 0.4097±0.036)** |
| `span_cuad_deberta_pretrain` | `train_span` | CUAD | Phase B step 1: NER pre-training on CUAD |
| `span_accord_deberta_phase_b` | `train_span` | ACCORD | **Phase B: + CUAD pre-train (std −42%)** |

Phase A+ flags (A20+A21+A12):
- **A12** — `--re-context-span`: in-span evidence text as third RE feature
- **A20** — staircase adaptive boost: `--re-comparison-boost 5.0 --re-boost-end 2.0 --re-boost-mid 3.5 --re-boost-adaptive-steps 1000`
- **A21** — `--label-smoothing 0.1`: prevents over-confident boundary negatives

---

## Pipeline steps

| Step | Flag name | Description | Requires |
|---|---|---|---|
| Train | `train` | Run the training script; save best checkpoint | — |
| Infer | `infer` | Extract triples from test split | Checkpoint |
| Verify | `verify` | LLM semantic verification (Ollama) | Inference JSONL |
| Build | `build` | Build KG graph JSON from triples | Inference JSONL |
| Triple eval | `triple` | NER F1 + Triple F1 from checkpoint | Checkpoint |
| RAG eval | `rag` | Graph RAG accuracy vs LLM-only baseline | KG JSON + Ollama |
| Compare | `compare` | Triple overlap F1 vs gold annotations | Inference JSONL |

Default `--steps`: `train,infer,build,triple,compare`
(verify and rag are omitted by default as they require Ollama)

Use `--steps all` as shorthand for `train,infer,verify,build,triple,rag,compare`.

---

## File naming convention

```
checkpoints/{train_script}_{attempt}_s{seed}[_n{N}][_e{N}][_d{dataset}]_best.pt

results/kg_{attempt}_s{seed}[suffix]_inference.jsonl      # raw predicted + gold triples
results/kg_{attempt}_s{seed}[suffix]_verified.jsonl       # after LLM verification
results/kg_{attempt}_s{seed}[suffix].json                 # built KG graph
results/graph_rag_{attempt}_s{seed}[suffix].json          # RAG evaluation results
results/kg_compare_{attempt}_s{seed}[suffix].json         # triple overlap F1 scores
results/triple_eval_{attempt}_s{seed}[suffix].json        # NER F1 + Triple F1 from eval step
results/pipeline_results.csv                              # unified table (all attempts × seeds)
```

The optional `[suffix]` segments are each present only when their flag is explicitly passed,
so every flag that affects the output is reflected in the filename:

| Segment | Flag | Effect on output |
|---|---|---|
| `_n{N}` | `--max-steps N` | Different training budget → different checkpoint |
| `_e{N}` | `--eval-every N` | Different save cadence → different checkpoint |
| `_d{name}` | `--force-dataset name` | Different training data → different checkpoint |

Segments are appended in that order. A production run (no overrides) has no suffix at all.

Examples:
```
# Production run (no overrides)
checkpoints/train_span_span_accord_deberta_aplus_s42_best.pt
results/kg_span_accord_deberta_aplus_s42_inference.jsonl

# Smoke test: --max-steps 2 --eval-every 1
checkpoints/train_span_span_accord_deberta_aplus_s42_n2_e1_best.pt
results/kg_span_accord_deberta_aplus_s42_n2_e1_inference.jsonl

# Cross-dataset smoke test: --max-steps 2 --eval-every 1 --force-dataset accord
checkpoints/train_stage2_scierc_bert_s42_n2_e1_daccord_best.pt
results/kg_scierc_bert_s42_n2_e1_daccord_inference.jsonl
```

---

## Phase B (CUAD transfer)

Phase B requires a CUAD pre-training checkpoint before ACCORD fine-tuning.
Run the two steps in order:

```bash
# Step 1: CUAD NER pre-training (~3500 steps, DeBERTa-large)
uv run python run_pipeline.py --attempt span_cuad_deberta_pretrain --seed 42 \
    --steps train

# Step 2: ACCORD fine-tuning from CUAD backbone
uv run python run_pipeline.py --attempt span_accord_deberta_phase_b --seed 42
```

The master script automatically injects `--pretrain-ckpt checkpoints/train_span_span_cuad_deberta_pretrain_s42_best.pt` when running Phase B.

---

## Reproducing all documented results + full evaluation

Requires Ollama running locally for the `verify` and `rag` steps.

```bash
# Step 1 — all 12 attempts at seed 42.
#
# Seed 42 is the only documented seed for every attempt except where noted below.
#
#   scierc_bert             morning_2026-04-11.md, morning_2026-04-12.md
#                           Stage 2 BIO baseline; seed 42 used throughout
#                           (morning_2026-04-14.md: "All experiments used seed 42").
#
#   multi_scierc_scibert    morning_2026-04-12.md, morning_2026-04-13.md
#   multi_conll04_bert      morning_2026-04-13.md, morning_2026-04-14.md
#   multi_ade_scibert       morning_2026-04-13.md, morning_2026-04-14.md
#                           BIO multi-dataset baselines run at seed 42 only. The 5-seed
#                           validation (42,123,456,7,13) in morning_2026-04-17.md was for
#                           train_span with --bio-weight, not train_multi — a different
#                           script and config; those results do not apply here.
#
#   span_scierc_scibert     morning_2026-04-14.md, morning_2026-04-15.md
#                           Span v10; seed 42 is the primary sweep seed
#                           (morning_2026-04-14.md: "All experiments used seed 42 except
#                           multi-seed validation (42, 123, 456)").
#                           Seeds 123, 456, 7, 13 reproduced in Step 2 below.
#
#   span_scierc_scibert_bio morning_2026-04-16.md, morning_2026-04-17.md, morning_2026-04-18.md
#                           BIO multi-task; seed 42 is the primary result seed.
#                           Seeds 123, 456, 7, 13 reproduced in Step 2 below.
#
#   span_accord_bert        morning_2026-04-19.md, morning_2026-04-21.md
#                           morning_2026-04-21.md shows seed 42 | 0.373 and seed 123 | 0.368,
#                           but that 2-seed run used flags (bio=0.1, neg=3.0, re_focal=2.0,
#                           conf=0.3) that are absent from this attempt's config in
#                           run_pipeline.py. Seed 123 is not reproducible with the current
#                           definition; seed 42 only.
#
#   span_accord_deberta     morning_2026-04-25.md, morning_2026-04-30.md,
#                           morning_2026-05-01.md, morning_2026-05-02.md
#                           8-seed (42–49) fully documented; morning_2026-05-01.md:
#                           "best DeBERTa-large checkpoint remains seed 42 (dev Triple=0.4288)".
#                           Seeds 43–49 reproduced in Step 2 below.
#
#   span_accord_deberta_a12 morning_2026-05-09.md, morning_2026-05-10.md
#                           A12 (--re-context-span) standalone was not run 8-seed. The 8-seed
#                           run in morning_2026-05-10.md was for A16+A12 combined (adds
#                           adaptive boost flags absent from this config); those results do
#                           not apply here. Seed 42 only.
#
#   span_accord_deberta_aplus  morning_2026-05-10.md – morning_2026-05-14.md
#                           8-seed (42–49); seeds 43–49 reproduced in Step 2 below.
#
#   span_cuad_deberta_pretrain morning_2026-05-14.md, morning_2026-05-16.md
#                           morning_2026-05-14.md: "CUAD Pre-train (3500 steps, seed=42)".
#                           Only seed 42 was ever run for the pretrain step; all Phase B
#                           fine-tunes in morning_2026-05-16.md used cuad_pretrain_s42_short.pt.
#
#   span_accord_deberta_phase_b  morning_2026-05-16.md, morning_2026-05-18.md, morning_2026-05-19.md
#                           8-seed fine-tune (seeds 42–49) from a single CUAD checkpoint
#                           (cuad_pretrain_s42_short.pt); morning_2026-05-16.md:
#                           mean=0.4096, std=0.0205. run_pipeline.py derives --pretrain-ckpt
#                           from the fine-tune seed, so seeds 43–49 would look for CUAD
#                           checkpoints that were never created. Not reproducible for seeds
#                           43–49 without a code change to run_pipeline.py.
#
# span_cuad_deberta_pretrain runs before span_accord_deberta_phase_b automatically.
uv run python run_pipeline.py \
    --attempt all \
    --seed 42 \
    --steps all \
    --ollama-url http://localhost:11434

# Step 2a — span_accord_deberta remaining seeds (43–49).
# morning_2026-04-30.md, morning_2026-05-01.md: 8-seed mean=0.3797 ± 0.0346.
# Per-seed dev: s42=0.4288, s43=0.380, s44=0.378, s45=0.415,
#              s46=0.3697, s47=0.3121, s48=0.3738, s49=0.3805
for seed in 43 44 45 46 47 48 49; do
    uv run python run_pipeline.py \
        --attempt span_accord_deberta \
        --seed $seed \
        --steps all \
        --ollama-url http://localhost:11434
done

# Step 2b — span_scierc_scibert remaining seeds (123, 456, 7, 13).
# morning_2026-04-15.md, morning_2026-04-18.md: 5-seed no-BIO baseline mean=0.353 ± 0.010.
# Per-seed dev: s42=0.343, s123=0.366, s456=0.356, s7=0.358, s13=0.343
for seed in 123 456 7 13; do
    uv run python run_pipeline.py \
        --attempt span_scierc_scibert \
        --seed $seed \
        --steps all \
        --ollama-url http://localhost:11434
done

# Step 2c — span_scierc_scibert_bio remaining seeds (123, 456, 7, 13).
# morning_2026-04-16.md, morning_2026-04-18.md: 5-seed mean=0.398 ± 0.002.
# Per-seed dev (log-verified): s42=0.398, s123=0.394, s456=0.400, s7=0.399, s13=0.397
for seed in 123 456 7 13; do
    uv run python run_pipeline.py \
        --attempt span_scierc_scibert_bio \
        --seed $seed \
        --steps all \
        --ollama-url http://localhost:11434
done

# Step 2d — span_accord_deberta_aplus remaining seeds (43–49).
# morning_2026-05-11.md, morning_2026-05-19.md: 8-seed mean=0.4097 ± 0.0356, peak 0.4746 (s47).
# Per-seed dev: s42=0.3969, s43=0.4330, s44=0.4257, s45=0.3542,
#              s46=0.4000, s47=0.4746, s48=0.4061, s49=0.3871
for seed in 43 44 45 46 47 48 49; do
    uv run python run_pipeline.py \
        --attempt span_accord_deberta_aplus \
        --seed $seed \
        --steps all \
        --ollama-url http://localhost:11434
done
```

`--steps all` runs every step (`train,infer,verify,build,triple,rag,compare`) in a single
pass. If a checkpoint already exists from a prior run, the `train` step is skipped
automatically — no redundant training.

---

## DGX Spark / Blackwell GPU notes

DeBERTa-large on the DGX GB10 (Blackwell architecture) requires:

```bash
export PYTORCH_JIT=0
export PYTORCH_NVFUSER_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
```

Use the `scripts/run_train_accord.sh` and `scripts/run_train_cuad.sh` wrappers for remote DGX runs — they inject these env vars automatically.

`run_pipeline.py` is designed for local runs where CUDA_VISIBLE_DEVICES and env vars are pre-set in the shell.

---

## Understanding Triple F1 results

### Smoke-test results are expected to be near zero

When running with `--max-steps 1` or `--max-steps 2` for smoke testing, the model
is essentially random (untrained), so Triple F1 will be ~0.000–0.001. This is expected
and correct — the smoke tests verify that code paths execute without errors, not that
the model produces good predictions.

### Production targets

| Attempt | Steps | Target Triple F1 (dev) | Notes |
|---|---|---|---|
| `span_accord_deberta_aplus` | 3500 | 0.4097 ± 0.036 | Phase A+ best (8 seeds) |
| `span_accord_deberta_phase_b` | 3500 | ~0.4097 with ±std −42% | Phase B best |
| `span_scierc_scibert_bio` | 3000 | ~0.55 (SciERC) | Standard benchmark |

### Typical F1 breakdown for Phase A+ (dev split)

- **NER F1**: ~0.60–0.65 — span boundary detection
- **RE F1** (gold spans): ~0.50–0.55 — relation classification given correct spans
- **Triple F1** (full pipeline): ~0.40–0.42 — combined; lower than RE F1 because span
  errors compound with relation errors

### Low test F1 vs dev F1

Test F1 is typically 5–10 points lower than dev F1 on ACCORD. This is expected:
the ACCORD dataset is small (~300 documents), and the model slightly overfits the dev
distribution. Evaluate the dev metric for model selection; report test for final scores.

---

## Unified results CSV

After any run that includes a `triple`, `rag`, or `compare` step, the pipeline appends
or updates a row in `results/pipeline_results.csv`:

| Column | Source |
|---|---|
| `attempt`, `seed`, `max_steps` | run parameters |
| `timestamp` | wall clock at pipeline end |
| `ner_f1`, `re_f1`, `triple_f1` | `results/triple_eval_*.json` |
| `triple_overlap_p/r/f1` | `results/kg_compare_*.json` |
| `rag_accuracy` | `results/graph_rag_*.json` (requires Ollama) |

Rows are upserted by `(attempt, seed, max_steps)` key — re-running a step updates
existing columns rather than creating duplicate rows.

---

## Eval module notes

`eval/triple_f1.py` auto-detects model type from the checkpoint:
- If `span_ner_head.*` keys are present → span evaluation via `eval/span_f1.py`
- Otherwise → BIO evaluation (NER F1 only accurate for SciERC; triple F1 works for all)

`eval/span_f1.py` can also be called standalone:
```bash
uv run python eval/span_f1.py \
    --checkpoint checkpoints/train_span_span_accord_deberta_aplus_s42_best.pt \
    --dataset accord --split test --seed 42
```

---

## Excluded train scripts

Several `train_*.py` files exist in the repo but are intentionally absent from
`EXPERIMENT_CONFIGS`. None of their functionality is duplicated by the listed
scripts — each was excluded for a distinct reason.

| Script | Why excluded |
|---|---|
| `train_stage2b.py` | Trains a BIO `BertKGExtractor` with an adversarial RealismCritic loss, but **requires a frozen Qwen-0.5B decoder at training time** to generate synthetic sentences. The pipeline has no Qwen dependency during training; the encoder architecture and eval path are otherwise compatible. |
| `train_stage2c.py` | Trains a **LoRA-tuned Qwen decoder** (REINFORCE against critic + triple-recovery reward). The BERT encoder is frozen by default at a prior checkpoint. The deliverable is the Qwen LoRA weights, which `inference_kg.py` does not use. No new `BertKGExtractor` checkpoint is produced. |
| `train_stage2d.py` | Fork of `train_stage2c.py` with an improved reward formula. Same exclusion reason: the BERT encoder is frozen; the LoRA decoder is the deliverable. |
| `train_stage2e.py` | Trains a BIO `BertKGExtractor` on gold + LoRA-generated paraphrase data. **Compatible with the KG pipeline** in principle, but excluded because: (1) it requires a pre-generated LoRA-synth JSONL (`data/stage2e_synth_v8.jsonl`) that itself depends on a trained `train_stage2c/d` LoRA decoder; (2) its gold-only mode (`--synth-jsonl ""`) is functionally identical to `scierc_bert`; (3) the experiment was superseded by `train_span.py`, which showed larger Triple F1 gains from span NER than from LoRA data augmentation. |
| `train.py`, `train_adversarial.py`, `train_gan.py`, `train_gumbel.py` | Stage 1 scripts — operate on random toy vectors with a `KGEncoder`/`KGDecoder` architecture. No BERT backbone; incompatible with `inference_kg.py`. |
| `train_pretrain_cooperative.py` | ELECTRA-style cooperative pre-training. Produces a **discriminator backbone** checkpoint, not a KG extractor. Used only as `--pretrain-ckpt` input to `train_span.py`, not as a standalone pipeline entry. |

## Round-trip KG comparison

The `compare` step computes triple overlap F1 between predicted and gold triples
from the inference JSONL (no LLM needed).

For a more thorough round-trip comparison (LLM re-extraction from predicted triples):

```bash
uv run python run_pipeline.py --attempt span_accord_deberta_aplus --seed 42 \
    --steps infer,verify,compare \
    --round-trip \
    --ollama-url http://localhost:11434
```

With `--round-trip`, the compare step will report on the LLM-verified triples
(from the verify step) and compare corrected triples against gold.
