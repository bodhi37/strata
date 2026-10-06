#!/bin/bash
# hillclimb-sc117.sh <config.json> <port> <tag> [ctx-list] [max-tokens]
#   launch a Strata config, wait for health, measure decode + prefill, save JSON, leave engine up.
set -uo pipefail
ROOT=/home/bodhi/models/strata
CFG=${1:?config}
PORT=${2:-8123}
TAG=${3:-run}
CTX=${4:-1024,4096,16384}
MT=${5:-256}
RES=$ROOT/bench/results
mkdir -p "$RES"
OUT=$RES/sc117-$TAG-$(date +%Y%m%dT%H%M%S).json
cd "$ROOT"

echo "== [hill] $TAG :: stop any engine =="
bash srv.sh stop >/dev/null 2>&1 || true
sleep 10

echo "== [hill] $TAG :: launch $CFG on :$PORT =="
bash srv.sh "$CFG" "$PORT" || { echo "== [hill] $TAG :: LAUNCH FAILED =="; tail -60 "$ROOT/logs/$(basename "$CFG" .json).log" 2>/dev/null; exit 1; }

echo "== [hill] $TAG :: warmup =="
curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"Say the single word: ready"}],"max_tokens":8,"reasoning_effort":"none"}' >/dev/null || true

echo "== [hill] $TAG :: throughput sweep $(date -Is) =="
venv/bin/python bench_endpoint.py --port "$PORT" --contexts "$CTX" --max-tokens "$MT" --effort none --out "$OUT" || echo "[hill] sweep failed"

echo "== [hill] $TAG :: decode stability (5x @1024) =="
for i in 1 2 3 4 5; do
  venv/bin/python bench_endpoint.py --port "$PORT" --contexts 1024 --max-tokens 192 --effort none \
     --out "$RES/sc117-$TAG-dec$i.json" 2>/dev/null | grep -E 'decode=' || true
done
echo "== [hill] $TAG :: done $(date -Is); engine left running on :$PORT =="