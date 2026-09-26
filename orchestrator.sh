#!/bin/bash
# orchestrator.sh - hands-off sequence for the quant bake-off:
#   1. wait for heretic_finish.sh (IQ3_XXS) to fully complete (its benchmarks land in bench/results)
#   2. finish_quant.sh IQ3_M  8102   (waits for the pull_quant.sh IQ3_M rsync to land the shards)
#   3. finish_quant.sh IQ4_XS 8103
# Each finish_quant stops the previous server first (one engine in RAM at a time).
# The winner is relaunched afterwards by hand (see REPORT.md).
set -uo pipefail
ROOT=/home/bodhi/models/strata
LOG=$ROOT/logs/orchestrator.log
exec >>"$LOG" 2>&1

echo "======== orchestrator start $(date -Is) ========"

# 1. IQ3_XXS pipeline
until grep -q "heretic_finish all done" "$ROOT/logs/finish.log" 2>/dev/null; do
  sleep 60
done
echo "[o] IQ3_XXS pipeline complete $(date -Is)"

# 2. IQ3_M
bash "$ROOT/finish_quant.sh" IQ3_M 8102
echo "[o] IQ3_M pass complete $(date -Is)"

# 3. IQ4_XS
bash "$ROOT/finish_quant.sh" IQ4_XS 8103
echo "[o] IQ4_XS pass complete $(date -Is)"

echo "======== orchestrator done $(date -Is) — all three quants benchmarked ========"