# LLM Synthetic Data Quality and Diversity for Low-Resource RE

Source lookup: `sources/papers_20260504_llm_synth_quality_diversity_low_resource_re.md`

## Trigger

CODE-ACCORD A8 document windows were neutral, and schema-aware Qwen
augmentation improved DeBERTa-large only marginally at 8 seeds:
`0.3887 +/- 0.0210` vs plain `0.3797 +/- 0.0346`. The mixed per-seed deltas
suggest the bottleneck is not "add any synthetic examples", but selecting
synthetic examples that add relation coverage without destabilizing strong
seeds.

## Relevant Papers

- S2ynRE (ACL 2023): uses LLM-generated domain-adaptive synthetic data plus
  two-stage self-training. Reported gains can be large in low-resource RE, but
  the useful part for ACCORD is the staged transition from synthetic support to
  gold supervision rather than naive mixing.
- ODDA (Findings ACL 2025): selects demonstrations from model behavior, applies
  relation/attribute constraints, then chooses globally diverse synthetic
  examples. This directly matches the ACCORD failure mode: the current 84
  examples help some seeds but are too narrow/noisy to move the mean clearly.
- TKRE / knowledge-guided RE pretraining (2025): combines explanation-driven
  synthetic data with schema-constrained generation and staged training. The
  actionable idea is relation-rationale text, not another encoder pretraining
  run.
- LLM + ASP JERE (2025): uses structured constraints around LLM extraction.
  Full pipeline replacement is too large for overnight, but constraint-based
  validation of generated triples is low-risk.

## Actionable Constraints For Next Experiment

1. Increase synthetic diversity across relation types and surface forms.
2. Keep exact head/tail preservation and relation-schema constraints.
3. Prefer targeted examples for low-recall relations (`selection`, `part-of`,
   `necessity`, `equal`, numeric comparisons) over generic replay.
4. Avoid another full 8-seed run until a 2-seed probe beats the current
   augmentation behavior on both a strong and weak seed.
5. Keep `PYTORCH_JIT=0` for Spark DeBERTa-large runs on GB10.

## Proposed Next Probe

Build a combined synthetic set from:

- `results/accord_llm_aug_schema_filtered_s42.jsonl`
- `results/accord_llm_aug_targeted_lowrecall_s43.jsonl`

Deduplicate by `(synth_sentence, head, tail, rel)`, inspect relation counts,
then run a 2-seed DeBERTa-large probe on seeds 43 and 44. These are useful
because seed 43 was neutral under the first augmentation set, while seed 44
improved clearly.
