#!/bin/bash
# pull_orca.sh - rsync OrcaRouter IQ4_XS 3-shard + mmproj + MTP from x64 staging to local.
#   nohup bash pull_orca.sh > logs/pull-orca.log 2>&1 &
# Source (x64): /mnt/1tbssd/ssh-downloads/aimodels/qwen38-fn-orcarouter-iq4xs/
# Dest (local): ~/models/qwen3.8-flash-next-orcarouter-uncensored-iq4xs/ + hardlinks into model/
set -uo pipefail
ROOT=/home/bodhi/models/strata
SRC=x64:/mnt/1tbssd/ssh-downloads/aimodels/qwen38-fn-orcarouter-iq4xs
DEST=/home/bodhi/models/qwen3.8-flash-next-orcarouter-uncensored-iq4xs
MDIR=$ROOT/model
LOG=$ROOT/logs/pull-orca.log
mkdir -p "$DEST" "$MDIR" "$ROOT/logs"
exec >>"$LOG" 2>&1

echo "======== pull_orca start $(date -Is) ========"
FILES=(
  "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"
  "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf"
  "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf"
  "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"
  "Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf"
)
# wait for x64 staging to have the shards (poller retries gate every 120s)
for F in "${FILES[@]:0:3}"; do
  tries=0
  until ssh x64 "test -s '/mnt/1tbssd/ssh-downloads/aimodels/qwen38-fn-orcarouter-iq4xs/$F'" 2>/dev/null; do
    tries=$((tries+1)); echo "[pull] waiting for x64:$F (attempt $tries) $(date -Is)"
    [ "$tries" -gt 1440 ] && { echo "[pull] giving up on $F"; exit 1; }
    sleep 60
  done
done
for F in "${FILES[@]}"; do
  echo "[pull] rsync $F $(date -Is)"
  until rsync -a --partial --info=progress2 "$SRC/$F" "$DEST/"; do sleep 15; done
  echo "[pull] done $F $(stat -c%s "$DEST/$F") bytes"
done
# hardlink shards into strata model dir (engine reads from model/)
for F in "${FILES[@]}"; do
  [ -s "$DEST/$F" ] && ln -f "$DEST/$F" "$MDIR/$F" 2>/dev/null || cp -f "$DEST/$F" "$MDIR/$F"
done
ls -lh "$DEST/" "$MDIR/"*Uncensored* 2>&1
echo "PULL_ORCA_DONE $(date -Is)"
