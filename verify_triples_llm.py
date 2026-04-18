"""
LLM-as-Verifier: Use on-premise Qwen3:32b to verify encoder-predicted triples.

For each predicted triple, asks the LLM:
"Given this sentence, is the relation (head, RELATION, tail) correct?"

This is Layer 2 of the KG quality pipeline:
  Layer 1: Confidence filtering (softmax threshold) — done in inference_kg.py
  Layer 2: LLM semantic verification — this script
  Layer 3: KG consistency check — todo

All processing runs on-premise (Ollama). No data leaves the local network.

Usage:
    uv run python verify_triples_llm.py \
        --input results/kg_inference_scierc_test.jsonl \
        --output results/kg_verified_scierc_test.jsonl \
        --min-triple-conf 0.3
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


VERIFY_PROMPT = """You are a scientific knowledge graph expert. Given a sentence from a scientific paper and a predicted relation triple, determine if the triple is correct.

Sentence: "{sentence}"

Predicted triple: ({head}, {relation}, {tail})

Is this triple correctly extracted from the sentence? Consider:
1. Are both entities actually mentioned in the sentence?
2. Does the stated relation accurately describe how they relate in the sentence?
3. Is the direction of the relation correct?

Answer with ONLY one word: Yes, No, or Uncertain."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input JSONL from inference_kg.py")
    p.add_argument("--output", required=True, help="Output JSONL with LLM verification")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-model", default="qwen3:32b")
    p.add_argument("--min-triple-conf", type=float, default=0.0,
                   help="Only verify triples with confidence >= this threshold. "
                        "Saves LLM calls on obviously bad triples.")
    p.add_argument("--max-docs", type=int, default=0, help="0 = all")
    return p.parse_args()


def call_ollama(ollama_url, model, prompt):
    """Call Ollama chat API with think:false for fast inference."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 10},
    })
    try:
        result = subprocess.run(
            ["curl", "-s", f"{ollama_url}/api/chat", "-d", payload],
            capture_output=True, text=True, timeout=60,
        )
        resp = json.loads(result.stdout)
        text = resp.get("message", {}).get("content", "").strip()
        return text
    except Exception as e:
        print(f"  Ollama error: {e}")
        return "Error"


def parse_verdict(response):
    """Parse LLM response to Yes/No/Uncertain."""
    r = response.lower().strip().rstrip(".").strip()
    if r.startswith("yes"):
        return "yes"
    elif r.startswith("no"):
        return "no"
    elif r.startswith("uncertain"):
        return "uncertain"
    return "unknown"


def main():
    args = parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    if args.max_docs > 0:
        records = records[:args.max_docs]

    total_triples = sum(len(r["predicted_triples"]) for r in records)
    eligible = sum(
        1 for r in records for t in r["predicted_triples"]
        if t["triple_conf"] >= args.min_triple_conf
    )
    print(f"=== LLM Triple Verification (On-Premise) ===")
    print(f"  Model: {args.ollama_model} via {args.ollama_url}")
    print(f"  Input: {args.input} ({len(records)} docs, {total_triples} triples)")
    print(f"  Min confidence: {args.min_triple_conf} → {eligible} triples to verify")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_verified = 0
    n_yes = n_no = n_uncertain = n_skip = 0
    t0 = time.time()

    with open(out_path, "w") as fout:
        for rec in records:
            verified_triples = []

            for triple in rec["predicted_triples"]:
                if triple["triple_conf"] < args.min_triple_conf:
                    triple["llm_verdict"] = "skipped"
                    verified_triples.append(triple)
                    n_skip += 1
                    continue

                prompt = VERIFY_PROMPT.format(
                    sentence=rec["sentence"],
                    head=triple["head_text"],
                    relation=triple["relation"],
                    tail=triple["tail_text"],
                )
                response = call_ollama(args.ollama_url, args.ollama_model, prompt)
                verdict = parse_verdict(response)

                triple["llm_verdict"] = verdict
                triple["llm_raw"] = response
                verified_triples.append(triple)

                if verdict == "yes":
                    n_yes += 1
                elif verdict == "no":
                    n_no += 1
                else:
                    n_uncertain += 1
                n_verified += 1

            rec["predicted_triples"] = verified_triples
            fout.write(json.dumps(rec) + "\n")

            if (rec["doc_id"] + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{rec['doc_id']+1}/{len(records)}] "
                      f"verified={n_verified} yes={n_yes} no={n_no} unc={n_uncertain} "
                      f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n=== Verification Complete ===")
    print(f"  Verified: {n_verified}, Skipped: {n_skip}")
    print(f"  Yes: {n_yes} ({n_yes/max(n_verified,1):.1%})")
    print(f"  No: {n_no} ({n_no/max(n_verified,1):.1%})")
    print(f"  Uncertain: {n_uncertain} ({n_uncertain/max(n_verified,1):.1%})")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(n_verified,1):.1f}s/triple)")
    print(f"  Output: {out_path}")

    # Evaluate: compare LLM-filtered vs raw vs gold
    with open(out_path) as f:
        verified_records = [json.loads(line) for line in f]

    has_gold = any(r["gold_triples"] for r in verified_records)
    if has_gold:
        print(f"\n  Quality comparison (with gold labels):")
        for label, filter_fn in [
            ("All predicted", lambda t: True),
            ("Confidence >= 0.5", lambda t: t["triple_conf"] >= 0.5),
            ("LLM = Yes", lambda t: t.get("llm_verdict") == "yes"),
            ("Conf >= 0.5 + LLM = Yes", lambda t: t["triple_conf"] >= 0.5 and t.get("llm_verdict") == "yes"),
        ]:
            tp = fp = fn = 0
            for rec in verified_records:
                pred_set = {
                    (tuple(t["head"]), tuple(t["tail"]), t["relation"])
                    for t in rec["predicted_triples"] if filter_fn(t)
                }
                gold_set = {
                    (tuple(t["head"]), tuple(t["tail"]), t["relation"])
                    for t in rec["gold_triples"]
                }
                tp += len(pred_set & gold_set)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"    {label:30s}: P={p:.3f} R={r:.3f} F1={f1:.3f} (kept={tp+fp})")


if __name__ == "__main__":
    main()
