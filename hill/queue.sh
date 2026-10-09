#!/usr/bin/env bash
# hill/queue.sh - run the R26 arm queue back to back.  One engine restart per arm
# (the endpoint is the box: 12 GB VRAM / 32 GB RAM cannot host a second engine).
# Every arm restarts the server on :8126, so user traffic in flight dies; that is
# the price of the hillclimb and the winner stays live at the end.
set -u
cd "$(dirname "$0")/.."
PY=python3
CTX=${CTX:-"4096"}
N=${N:-3}

run() {
  local tag=$1; shift
  $PY hill/hc.py --tag "$tag" --cfg "hill/cfg-$tag.json" --port 8126 \
      --ctxs "$CTX" --n "$N" 2>&1 | tee -a "logs/hc2-$tag.log"
}

$PY hill/mkcfg.py spec8     --set spec=8
$PY hill/mkcfg.py spec8mp70 --set spec=8 --set spec-min-p=0.7
$PY hill/mkcfg.py spec8rs4  --set spec=8 --set env=STRATA_RSPLIT=4
$PY hill/mkcfg.py spec8c17  --set spec=8 --set expert-cache=1700 --set vram-reserve-mib=300

run spec8
run spec8mp70
run spec8rs4
run spec8c17
echo "QUEUE DONE"
