#!/bin/bash
# pull_quant.sh <QUANT> - rsync the 3 per-quant shards (1/4/5) from x64 once aria2 has finished them.
#   nohup bash pull_quant.sh IQ3_M  > logs/pull-iq3_m.log 2>&1 &
set -uo pipefail
QUANT=${1:?quant}
ROOT=/home/bodhi/models/strata
RP=/mnt/1tbssd/ssh-downloads/aimodels/qwen38-fn-heretic2-$(echo "$QUANT" | tr 'A-Z_' 'a-z-')/$QUANT   # path on x64
SRC=x64:$RP
D=$ROOT/model
B=Qwen3.8-Flash-Next-heretic-2-$QUANT
declare -A WANT
case "$QUANT" in
  IQ3_M)  WANT[00001]=10945760;   WANT[00004]=47853314784; WANT[00005]=14813686368 ;;
  IQ4_XS) WANT[00001]=10945760;   WANT[00004]=47566147392; WANT[00005]=22650600960 ;;
  *) echo "unknown quant"; exit 1 ;;
esac
for N in 00001 00005 00004; do
  F="$D/$B-$N-of-00005.gguf"
  until [ "$(ssh x64 "stat -c%s '$RP/$B-$N-of-00005.gguf' 2>/dev/null" 2>/dev/null)" = "${WANT[$N]}" ]; do
    sleep 30
  done
  # let any other rsync already writing this file finish first
  while pgrep -f "rsync.*$B-$N-of-00005" >/dev/null; do sleep 30; done
  if [ -f "$F" ] && [ "$(stat -c%s "$F")" = "${WANT[$N]}" ]; then
    echo "already present $N $(date -Is)"; continue
  fi
  until rsync -a --partial --info=progress2 "$SRC/$B-$N-of-00005.gguf" "$D/"; do sleep 15; done
  echo "pulled $N $(date -Is)"
done
echo "PULL_${QUANT}_DONE"