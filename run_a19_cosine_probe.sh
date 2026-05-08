#!/bin/bash
# A19: Cosine Curriculum Boost 2-seed probe
# --re-boost-decay-cosine changes linear decay to cosine decay
# Cosine decays slower initially (better early signal), faster near end
# Compare vs A15 linear decay (same seeds 43,44)
# Run this ONLY after A15 2-seed probe result is available for comparison

cd ~/autoresearch
source $HOME/.local/bin/env 2>/dev/null || true

LOG_DIR="results/a19_cosine_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary.log"

echo "A19 Cosine Curriculum Boost (5.0x -> 2.0x cosine) 2-seed probe" | tee "$SUMMARY_LOG"
echo "Config: DeBERTa-large, --re-comparison-boost 5.0 --re-boost-end 2.0 --re-boost-decay-cosine, 3500 steps, seeds 43,44" | tee -a "$SUMMARY_LOG"
echo "Compare A15 linear (same seeds, same config without --re-boost-decay-cosine)" | tee -a "$SUMMARY_LOG"
echo "Started: $(date)" | tee -a "$SUMMARY_LOG"

for SEED in 43 44; do
    echo "=== Seed $SEED ===" | tee -a "$SUMMARY_LOG"
    PYTORCH_JIT=0 uv run python train_span.py \
      --dataset accord \
      --model-name microsoft/deberta-large \
      --bio-weight 0.1 \
      --neg-sample-ratio 3.0 \
      --re-comparison-boost 5.0 \
      --re-boost-end 2.0 \
      --re-boost-decay-cosine \
      --max-steps 3500 \
      --seed $SEED \
      --save-best-to checkpoints/deberta_large_accord_s${SEED}_a19_cosine_best.pt \
      > "$LOG_DIR/seed_${SEED}.log" 2>&1
    TRIPLE=$(grep -A1 "BEST DEV" "$LOG_DIR/seed_${SEED}.log" | grep -oP "Triple=\K[0-9.]+" | tail -1)
    NER=$(grep -A1 "BEST DEV" "$LOG_DIR/seed_${SEED}.log" | grep -oP "NER=\K[0-9.]+" | tail -1)
    echo "  Seed $SEED: Triple=$TRIPLE NER=$NER" | tee -a "$SUMMARY_LOG"
done
echo "=== DONE $(date) ===" | tee -a "$SUMMARY_LOG"
