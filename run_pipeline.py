"""
run_pipeline.py — Master KG Research Pipeline Runner

Orchestrates training → inference → [verification] → KG construction → evaluation
for all distinct BertKGExtractor experiment configurations.

All paths are relative to the autoresearch/ directory.

Usage:
    python run_pipeline.py --list-attempts
    python run_pipeline.py --attempt scierc_bert --seed 42
    python run_pipeline.py --attempt accord_deberta_aplus --seed 42
    python run_pipeline.py --attempt accord_deberta_aplus --seed 42 --steps train,infer,triple
    python run_pipeline.py --attempt all --seed 42 --dry-run

    # Phase B requires CUAD pretrain first:
    python run_pipeline.py --attempt span_cuad_deberta_pretrain --seed 42
    python run_pipeline.py --attempt span_accord_deberta_phase_b --seed 42

    # With Ollama LLM verification and RAG evaluation:
    python run_pipeline.py --attempt accord_deberta_aplus --seed 42 \\
        --steps train,infer,verify,build,triple,rag,compare \\
        --ollama-url http://localhost:11434
"""
import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

# Prefer the venv Python (has torch/transformers) over the invoking interpreter.
_VENV_PY_WIN = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
_VENV_PY_UNIX = Path(__file__).parent / ".venv" / "bin" / "python"
_PYTHON = (
    str(_VENV_PY_WIN) if _VENV_PY_WIN.exists() else
    str(_VENV_PY_UNIX) if _VENV_PY_UNIX.exists() else
    sys.executable
)


# ── A20+A21+A12 production baseline flags (Phase A+) ─────────────────────────
# Source: scripts/run_train_accord.sh
_A20_A21_A12 = [
    "--re-comparison-boost", "5.0",
    "--re-boost-end", "2.0",
    "--re-boost-mid", "3.5",
    "--re-boost-adaptive-threshold", "0.35",
    "--re-boost-adaptive-threshold2", "0.40",
    "--re-boost-adaptive-steps", "1000",
    "--label-smoothing", "0.1",
    "--re-context-span",
    "--neg-sample-ratio", "3.0",
]


# ── Experiment configurations ─────────────────────────────────────────────────
#
# Keys: attempt_name (used in checkpoint and artifact paths)
# Each entry specifies the training script, dataset, backbone model, CLI flags,
# and optional pretrain dependency.
#
# Checkpoints: checkpoints/{train_script}_{attempt}_s{seed}_best.pt
# Artifacts:   results/kg_{attempt}_s{seed}_{stage}.jsonl|.json
EXPERIMENT_CONFIGS = {

    # ── train_stage2 (BIO supervised, SciERC) ────────────────────────────────
    "scierc_bert": {
        "train_script": "train_stage2",
        "dataset": "scierc",
        "model_name": "bert-base-uncased",
        "extra_args": [
            "--max-steps", "1000",
            "--warmup-steps", "100",
            "--eval-every", "100",
        ],
        "description": "BERT-base BIO supervised, SciERC (Stage 2 baseline) [reports: morning_2026-04-11.md, morning_2026-04-12.md]",
    },

    # ── train_multi (BIO multi-dataset) ──────────────────────────────────────
    "multi_scierc_scibert": {
        "train_script": "train_multi",
        "dataset": "scierc",
        "model_name": "allenai/scibert_scivocab_uncased",
        "extra_args": [
            "--max-steps", "1500",
            "--warmup-steps", "250",
            "--eval-every", "100",
        ],
        "description": "SciBERT BIO multi-dataset, SciERC [reports: morning_2026-04-12.md, morning_2026-04-13.md]",
    },
    "multi_conll04_bert": {
        "train_script": "train_multi",
        "dataset": "conll04",
        "model_name": "bert-base-uncased",
        "extra_args": [
            "--max-steps", "1500",
            "--warmup-steps", "250",
            "--eval-every", "100",
        ],
        "description": "BERT-base BIO multi-dataset, CoNLL04 [reports: morning_2026-04-13.md, morning_2026-04-14.md]",
    },
    "multi_ade_scibert": {
        "train_script": "train_multi",
        "dataset": "ade",
        "model_name": "allenai/scibert_scivocab_uncased",
        "extra_args": [
            "--max-steps", "1500",
            "--warmup-steps", "250",
            "--eval-every", "100",
        ],
        "description": "SciBERT BIO multi-dataset, ADE [reports: morning_2026-04-13.md, morning_2026-04-14.md]",
    },

    # ── train_span (span NER, SciERC) ─────────────────────────────────────────
    "span_scierc_scibert": {
        "train_script": "train_span",
        "dataset": "scierc",
        "model_name": "allenai/scibert_scivocab_uncased",
        "extra_args": [
            "--max-steps", "3000",
            "--warmup-steps", "250",
            "--eval-every", "200",
        ],
        "description": "Phase 12: Span NER baseline, SciERC, SciBERT [reports: morning_2026-04-14.md, morning_2026-04-15.md, morning_2026-04-16.md, morning_2026-04-17.md]",
    },
    "span_scierc_scibert_bio": {
        "train_script": "train_span",
        "dataset": "scierc",
        "model_name": "allenai/scibert_scivocab_uncased",
        "extra_args": [
            "--max-steps", "3000",
            "--warmup-steps", "250",
            "--eval-every", "200",
            "--bio-weight", "0.1",
        ],
        "description": "Phase 14: Span+BIO multi-task, SciERC, SciBERT [reports: morning_2026-04-16.md, morning_2026-04-17.md, morning_2026-04-18.md]",
    },

    # ── train_span (span NER, ACCORD ablations) ───────────────────────────────
    "span_accord_bert": {
        "train_script": "train_span",
        "dataset": "accord",
        "model_name": "bert-base-uncased",
        "extra_args": [
            "--max-steps", "3500",
            "--warmup-steps", "250",
            "--eval-every", "200",
        ],
        "description": "Phase A early: Span NER, ACCORD, BERT-base [reports: morning_2026-04-19.md, morning_2026-04-20.md, morning_2026-04-21.md]",
    },
    "span_accord_deberta": {
        "train_script": "train_span",
        "dataset": "accord",
        "model_name": "microsoft/deberta-large",
        "extra_args": [
            "--max-steps", "3500",
            "--warmup-steps", "250",
            "--eval-every", "200",
        ],
        "description": "Phase A: Span NER, ACCORD, DeBERTa-large (no A-flags) [reports: morning_2026-04-25.md, morning_2026-04-30.md, morning_2026-05-01.md, morning_2026-05-02.md]",
    },
    "span_accord_deberta_a12": {
        "train_script": "train_span",
        "dataset": "accord",
        "model_name": "microsoft/deberta-large",
        "extra_args": [
            "--max-steps", "3500",
            "--warmup-steps", "250",
            "--eval-every", "200",
            "--re-context-span",
        ],
        "description": "Phase A12: DeBERTa-large + context-span RE, ACCORD [reports: morning_2026-05-09.md, morning_2026-05-10.md]",
    },
    "span_accord_deberta_aplus": {
        "train_script": "train_span",
        "dataset": "accord",
        "model_name": "microsoft/deberta-large",
        "extra_args": [
            "--max-steps", "3500",
            "--warmup-steps", "250",
            "--eval-every", "200",
        ] + _A20_A21_A12,
        "description": "Phase A+: DeBERTa-large + A20+A21+A12, ACCORD (best: dev=0.4097±0.036) [reports: morning_2026-05-10.md, morning_2026-05-11.md, morning_2026-05-12.md, morning_2026-05-13.md, morning_2026-05-14.md]",
    },

    # ── train_span (CUAD NER pre-training, Phase B step 1) ───────────────────
    "span_cuad_deberta_pretrain": {
        "train_script": "train_span",
        "dataset": "cuad",
        "model_name": "microsoft/deberta-large",
        "extra_args": [
            "--max-steps", "3500",
            "--warmup-steps", "250",
            "--eval-every", "200",
            "--primary-metric", "ner_f1",
        ],
        "description": "Phase B step 1: DeBERTa-large NER pre-training on CUAD [reports: morning_2026-05-14.md, morning_2026-05-16.md]",
    },

    # ── train_span (ACCORD fine-tune from CUAD backbone, Phase B step 2) ─────
    "span_accord_deberta_phase_b": {
        "train_script": "train_span",
        "dataset": "accord",
        "model_name": "microsoft/deberta-large",
        "extra_args": [
            "--max-steps", "3500",
            "--warmup-steps", "250",
            "--eval-every", "200",
        ] + _A20_A21_A12,
        "requires_pretrain": "span_cuad_deberta_pretrain",
        "description": "Phase B: DeBERTa-large + A20+A21+A12 + CUAD pre-train, ACCORD (std -42%) [reports: morning_2026-05-16.md, morning_2026-05-18.md, morning_2026-05-19.md]",
    },
}


# ── Path helpers ──────────────────────────────────────────────────────────────

def _name_suffix(max_steps=None, eval_every=None, force_dataset=None) -> str:
    """Build the deterministic filename suffix from every flag that affects output.

    Segment meanings (each only present when the flag was explicitly overridden):
      _n{N}     --max-steps N        different training budget → different checkpoint
      _e{N}     --eval-every N       different save cadence    → different checkpoint
      _d{name}  --force-dataset name different training data   → different checkpoint
    """
    s = ""
    if max_steps is not None:
        s += f"_n{max_steps}"
    if eval_every is not None:
        s += f"_e{eval_every}"
    if force_dataset is not None:
        s += f"_d{force_dataset}"
    return s


def checkpoint_path(attempt: str, seed: int, max_steps=None, eval_every=None,
                    force_dataset=None) -> Path:
    cfg = EXPERIMENT_CONFIGS[attempt]
    suffix = _name_suffix(max_steps, eval_every, force_dataset)
    return Path(f"checkpoints/{cfg['train_script']}_{attempt}_s{seed}{suffix}_best.pt")


def artifact_paths(attempt: str, seed: int, max_steps=None, eval_every=None,
                   force_dataset=None) -> dict:
    suffix = _name_suffix(max_steps, eval_every, force_dataset)
    base = f"{attempt}_s{seed}{suffix}"
    return {
        "inference":   Path(f"results/kg_{base}_inference.jsonl"),
        "verified":    Path(f"results/kg_{base}_verified.jsonl"),
        "kg":          Path(f"results/kg_{base}.json"),
        "rag":         Path(f"results/graph_rag_{base}.json"),
        "compare":     Path(f"results/kg_compare_{base}.json"),
        "triple_eval": Path(f"results/triple_eval_{base}.json"),
    }


# ── Command helpers ───────────────────────────────────────────────────────────

def _py(*args):
    return [_PYTHON] + [str(a) for a in args]


def _py_mod(module, *args):
    return [_PYTHON, "-m", module] + [str(a) for a in args]


def run_cmd(cmd: list, dry_run: bool = False) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return 0
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(cmd, env=env).returncode


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_train(cfg, attempt, seed, checkpoint, dry_run, device,
               max_steps_override=None, eval_every_override=None, force=False,
               force_dataset=None):
    """5a: Train the model and save the best checkpoint."""
    if not dry_run and not force and checkpoint.exists():
        print(f"[skip] checkpoint already exists: {checkpoint}")
        return 0

    script = f"{cfg['train_script']}.py"
    extra = list(cfg["extra_args"])

    def _override_arg(args_list, flag, value):
        """Remove existing flag+value pair from args_list, then append new value."""
        clean, skip = [], False
        for tok in args_list:
            if skip:
                skip = False
                continue
            if tok == flag:
                skip = True
                continue
            clean.append(tok)
        return clean + [flag, str(value)]

    if max_steps_override is not None:
        extra = _override_arg(extra, "--max-steps", max_steps_override)
    if eval_every_override is not None:
        extra = _override_arg(extra, "--eval-every", eval_every_override)
    args = [
        "--dataset", cfg["dataset"],
        "--model-name", cfg["model_name"],
        "--seed", str(seed),
        "--save-best-to", str(checkpoint),
    ] + extra

    if "requires_pretrain" in cfg:
        pretrain_name = cfg["requires_pretrain"]
        pretrain_ckpt = checkpoint_path(pretrain_name, seed, max_steps_override,
                                        eval_every_override, force_dataset)
        if not dry_run and not pretrain_ckpt.exists():
            print(f"[error] Phase B requires pretrain checkpoint: {pretrain_ckpt}")
            print(f"  Run --attempt {pretrain_name} --seed {seed} first.")
            return 1
        if "--pretrain-ckpt" not in cfg["extra_args"]:
            args += ["--pretrain-ckpt", str(pretrain_ckpt)]

    if device:
        args += ["--device", device]

    return run_cmd(_py(script, *args), dry_run=dry_run)


def step_infer(cfg, attempt, seed, checkpoint, artifacts, dry_run, device):
    """5b.1: Inference — extract predicted triples + gold triples."""
    args = [
        "--checkpoint", str(checkpoint),
        "--dataset", cfg["dataset"],
        "--model-name", cfg["model_name"],
        "--split", "test",
        "--seed", str(seed),
        "--out-jsonl", str(artifacts["inference"]),
    ]
    if device:
        args += ["--device", device]
    return run_cmd(_py("inference_kg.py", *args), dry_run=dry_run)


def step_verify(cfg, attempt, seed, artifacts, dry_run, ollama_url):
    """5b.2 (optional): LLM verification of predicted triples via Ollama."""
    args = [
        "--input", str(artifacts["inference"]),
        "--output", str(artifacts["verified"]),
        "--ollama-url", ollama_url,
        "--mode", "correct",
    ]
    return run_cmd(_py("verify_triples_llm.py", *args), dry_run=dry_run)


def step_build_kg(artifacts, dry_run, use_verified=False):
    """5b.3: Build the KG graph JSON from triples."""
    inp = artifacts["verified"] if (use_verified and artifacts["verified"].exists()) \
        else artifacts["inference"]
    # Use "confidence" filter when no LLM verify ran (default "verified" would
    # silently produce an empty KG because llm_verdict is absent from the JSONL).
    filter_mode = "verified" if use_verified else "confidence"
    args = [
        "--input", str(inp),
        "--output", str(artifacts["kg"]),
        "--filter-mode", filter_mode,
    ]
    return run_cmd(_py("build_kg.py", *args), dry_run=dry_run)


def step_eval_triple(cfg, attempt, seed, checkpoint, artifacts, dry_run, device):
    """5c: Evaluate checkpoint Triple F1 (span or BIO, auto-detected)."""
    args = [
        "--checkpoint", str(checkpoint),
        "--dataset", cfg["dataset"],
        "--model-name", cfg["model_name"],
        "--split", "test",
        "--seed", str(seed),
        "--out-json", str(artifacts["triple_eval"]),
    ]
    if device:
        args += ["--device", device]
    return run_cmd(_py_mod("eval.triple_f1", *args), dry_run=dry_run)


def step_eval_rag(artifacts, dry_run, ollama_url):
    """5d: Graph RAG evaluation (requires Ollama)."""
    args = [
        "--kg", str(artifacts["kg"]),
        "--gold-jsonl", str(artifacts["inference"]),
        "--ollama-url", ollama_url,
        "--output", str(artifacts["rag"]),
    ]
    return run_cmd(_py("eval_graph_rag.py", *args), dry_run=dry_run)


def step_compare_kgs(attempt, seed, artifacts, dry_run, round_trip=False):
    """5e: Triple overlap F1 between predicted and gold KG."""
    if dry_run:
        print(f"[dry-run] compare KGs from {artifacts['inference']}")
        return 0

    infer_path = artifacts["inference"]
    if not infer_path.exists():
        print(f"[skip] compare — inference file not found: {infer_path}")
        return 1

    with open(infer_path) as f:
        records = [json.loads(line) for line in f]

    def _normalize(text):
        return text.lower().strip()

    tp = fp = fn = 0
    for rec in records:
        pred = {
            (_normalize(t["head_text"]), _normalize(t["tail_text"]), t["relation"])
            for t in rec.get("predicted_triples", [])
        }
        gold = {
            (_normalize(t["head_text"]), _normalize(t["tail_text"]), t["relation"])
            for t in rec.get("gold_triples", [])
        }
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    result = {
        "attempt": attempt,
        "seed": seed,
        "triple_overlap_p": round(p, 4),
        "triple_overlap_r": round(r, 4),
        "triple_overlap_f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "n_docs": len(records),
    }

    print(f"\n=== KG Comparison: {attempt}  seed={seed} ===")
    print(f"  Triple Overlap  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    print(f"  (tp={tp}  fp={fp}  fn={fn}  n_docs={len(records)})")

    compare_path = artifacts["compare"]
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    with open(compare_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {compare_path}")

    if round_trip:
        verified = artifacts["verified"]
        if not verified.exists():
            print("  [warn] --round-trip requires verified JSONL; run --steps verify first.")
        else:
            print(f"  [round-trip] LLM-verified file available: {verified}")
            print("  Re-run compare step against verified triples for full round-trip score.")

    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Master KG Research Pipeline Runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--attempt", default=None,
                   help="Attempt name (see --list-attempts). Use 'all' to run every config.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--steps",
        default="train,infer,build,triple,compare",
        help=(
            "Comma-separated pipeline steps. "
            "Available: train, infer, verify, build, triple, rag, compare. "
            "Default omits verify/rag (require Ollama). "
            "Use 'train,infer,verify,build,triple,rag,compare' for full pipeline."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    p.add_argument("--list-attempts", action="store_true",
                   help="List all available attempt configs and exit.")
    p.add_argument("--ollama-url", default="http://localhost:11434",
                   help="Ollama endpoint URL for verify and rag steps.")
    p.add_argument("--round-trip", action="store_true",
                   help="Enable round-trip KG comparison in compare step (requires --steps verify).")
    p.add_argument("--device", default=None,
                   help="Torch device override (default: auto-detect cuda/cpu).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override --max-steps for every attempt (e.g. 1 or 2 for smoke tests).")
    p.add_argument("--eval-every", type=int, default=None,
                   help="Override --eval-every for every attempt (useful with --max-steps for smoke tests).")
    p.add_argument("--force", action="store_true",
                   help="Re-run train even if the checkpoint already exists.")
    p.add_argument("--force-dataset", default=None,
                   help="Override the dataset for every attempt (e.g. 'accord'). "
                        "Useful for smoke-testing all configs against one dataset.")
    args = p.parse_args()

    if args.list_attempts:
        _list_attempts()
        return 0

    if args.attempt is None:
        p.error("--attempt is required (or use --list-attempts)")

    attempts = list(EXPERIMENT_CONFIGS.keys()) if args.attempt == "all" \
        else [args.attempt]

    for name in attempts:
        if name not in EXPERIMENT_CONFIGS:
            print(f"[error] Unknown attempt: {name!r}. Use --list-attempts.")
            sys.exit(1)

    steps = {s.strip() for s in args.steps.split(",")}
    valid_steps = {"train", "infer", "verify", "build", "triple", "rag", "compare"}
    unknown_steps = steps - valid_steps
    if unknown_steps:
        p.error(f"Unknown steps: {unknown_steps}. Valid: {sorted(valid_steps)}")

    for attempt in attempts:
        _run_attempt(attempt, args.seed, steps, args)

    print("\nAll done.")
    return 0


_CSV_COLUMNS = [
    "attempt", "seed", "max_steps", "timestamp",
    "ner_f1", "re_f1", "triple_f1",
    "triple_overlap_p", "triple_overlap_r", "triple_overlap_f1",
    "rag_accuracy",
]


def _update_csv_from_artifacts(attempt, seed, max_steps, arts, csv_path: Path):
    """Upsert a results row into pipeline_results.csv by merging artifact JSONs."""
    row = {
        "attempt": attempt,
        "seed": str(seed),
        "max_steps": str(max_steps) if max_steps is not None else "",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    triple_eval_path = arts.get("triple_eval")
    if triple_eval_path and triple_eval_path.exists():
        with open(triple_eval_path) as f:
            te = json.load(f)
        for k in ("ner_f1", "re_f1", "triple_f1"):
            if k in te:
                row[k] = te[k]

    compare_path = arts.get("compare")
    if compare_path and compare_path.exists():
        with open(compare_path) as f:
            cmp = json.load(f)
        for k in ("triple_overlap_p", "triple_overlap_r", "triple_overlap_f1"):
            if k in cmp:
                row[k] = cmp[k]

    rag_path = arts.get("rag")
    if rag_path and rag_path.exists():
        with open(rag_path) as f:
            rag = json.load(f)
        for k in ("accuracy", "rag_accuracy", "f1", "score"):
            if k in rag:
                row["rag_accuracy"] = rag[k]
                break

    key = (str(attempt), str(seed), str(max_steps) if max_steps is not None else "")
    rows = []
    found = False
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r_key = (r.get("attempt", ""), r.get("seed", ""), r.get("max_steps", ""))
                if r_key == key:
                    found = True
                    for col in _CSV_COLUMNS:
                        if col in row and str(row[col]) != "":
                            r[col] = str(row[col])
                    rows.append(r)
                else:
                    rows.append(r)

    if not found:
        rows.append({col: str(row.get(col, "")) for col in _CSV_COLUMNS})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] Updated {csv_path}")


def _list_attempts():
    col = 42
    print(f"\n{'Attempt':42s}  {'Script':16s}  {'Dataset':10s}  Description")
    print("-" * 110)
    for name, cfg in EXPERIMENT_CONFIGS.items():
        note = " *pretrain required*" if "requires_pretrain" in cfg else ""
        print(f"{name:{col}s}  {cfg['train_script']:16s}  {cfg['dataset']:10s}  "
              f"{cfg['description']}{note}")
    print(f"\n  * span_accord_deberta_phase_b requires span_cuad_deberta_pretrain "
          f"checkpoint first.")


def _run_attempt(attempt, seed, steps, args):
    cfg = dict(EXPERIMENT_CONFIGS[attempt])   # shallow copy — safe to mutate
    fd = getattr(args, "force_dataset", None)
    if fd:
        cfg["dataset"] = fd
    ckpt = checkpoint_path(attempt, seed, args.max_steps, args.eval_every, fd)
    arts = artifact_paths(attempt, seed, args.max_steps, args.eval_every, fd)

    print(f"\n{'='*64}")
    print(f"ATTEMPT : {attempt}")
    print(f"DESC    : {cfg['description']}")
    print(f"SEED    : {seed}")
    print(f"STEPS   : {', '.join(sorted(steps))}")
    print(f"CKPT    : {ckpt}")
    print(f"{'='*64}")

    use_verify = "verify" in steps

    if "train" in steps:
        rc = step_train(cfg, attempt, seed, ckpt, args.dry_run, args.device,
                        max_steps_override=args.max_steps,
                        eval_every_override=args.eval_every,
                        force=args.force,
                        force_dataset=fd)
        if rc != 0:
            print(f"[warn] train failed (rc={rc}); subsequent steps will skip if checkpoint/data missing")

    if "infer" in steps:
        if not args.dry_run and not ckpt.exists():
            print(f"[skip] infer — checkpoint not found: {ckpt}")
        else:
            rc = step_infer(cfg, attempt, seed, ckpt, arts, args.dry_run, args.device)
            if rc != 0:
                print(f"[warn] infer failed (rc={rc}); continuing")

    if "verify" in steps:
        if not args.dry_run and not arts["inference"].exists():
            print(f"[skip] verify — inference file not found")
        else:
            rc = step_verify(cfg, attempt, seed, arts, args.dry_run, args.ollama_url)
            if rc != 0:
                print(f"[warn] verify failed (rc={rc}); continuing")

    if "build" in steps:
        if not args.dry_run and not arts["inference"].exists():
            print(f"[skip] build — inference file not found")
        else:
            rc = step_build_kg(arts, args.dry_run, use_verified=use_verify)
            if rc != 0:
                print(f"[warn] build failed (rc={rc}); continuing")

    if "triple" in steps:
        if not args.dry_run and not ckpt.exists():
            print(f"[skip] triple eval — checkpoint not found: {ckpt}")
        else:
            rc = step_eval_triple(cfg, attempt, seed, ckpt, arts, args.dry_run, args.device)
            if rc != 0:
                print(f"[warn] triple eval failed (rc={rc}); continuing")

    if "rag" in steps:
        if not args.dry_run and not arts["kg"].exists():
            print(f"[skip] rag eval — KG not found: {arts['kg']}")
        else:
            rc = step_eval_rag(arts, args.dry_run, args.ollama_url)
            if rc != 0:
                print(f"[warn] rag eval failed (rc={rc}); continuing")

    if "compare" in steps:
        step_compare_kgs(attempt, seed, arts, args.dry_run, round_trip=args.round_trip)

    if {"triple", "rag", "compare"} & steps and not args.dry_run:
        _update_csv_from_artifacts(
            attempt, seed, args.max_steps, arts,
            Path("results/pipeline_results.csv"),
        )


if __name__ == "__main__":
    sys.exit(main())
