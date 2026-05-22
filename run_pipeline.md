# run_pipeline.py — Master KG Research Pipeline

Automates the full sequence: train → infer → [verify] → build KG → evaluate
for every distinct `BertKGExtractor` experiment configuration in this project.

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

---

## File naming convention

```
checkpoints/{train_script}_{attempt}_s{seed}_best.pt

results/kg_{attempt}_s{seed}_inference.jsonl   # raw predicted + gold triples
results/kg_{attempt}_s{seed}_verified.jsonl    # after LLM verification
results/kg_{attempt}_s{seed}.json              # built KG graph
results/graph_rag_{attempt}_s{seed}.json       # RAG evaluation results
results/kg_compare_{attempt}_s{seed}.json      # triple overlap F1 scores
```

Examples:
```
checkpoints/train_span_span_accord_deberta_aplus_s42_best.pt
results/kg_span_accord_deberta_aplus_s42_inference.jsonl
results/kg_span_accord_deberta_aplus_s42.json
results/kg_compare_span_accord_deberta_aplus_s42.json
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

## Multi-seed evaluation (8 seeds, matching Phase A+ protocol)

```bash
for seed in 42 47 51 55 57 61 67 71; do
    uv run python run_pipeline.py \
        --attempt span_accord_deberta_aplus \
        --seed $seed \
        --steps train,triple
done
```

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
