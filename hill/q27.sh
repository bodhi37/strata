#!/usr/bin/env bash
# R27 arm queue: lookahead prefetch + VRAM-resident skip A/B
cd "$(dirname "$0")/.."
for tag in "$@"; do
  python3 hill/hc.py --tag "$tag" --cfg "hill/cfg-$tag.json" --port 8126 --ctxs "${CTX:-4096}" --n "${N:-3}" 2>&1 | tee -a "logs/hc2-$tag.log" | grep -E "sample|decode|error|Error" 
done
echo QUEUE DONE
