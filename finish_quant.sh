#!/bin/bash
# finish_quant.sh <QUANT> <PORT> - bring up one spiritfather/heretic-2 quant under Strata and measure it.
#
#   bash finish_quant.sh IQ3_M 8102
#   bash finish_quant.sh IQ4_XS 8103
#
# Reuses the artifacts shared by every quant of this base model (verified byte-identical by HF
# LFS oid): shard 2 (native head tensor) and shard 3 (the 54.4 GB Q8_0 PLE table, requantized
# once to ple/ple-iq4nl.gguf), plus the shared MTP draft runtime in mtp/rt.
# Per-quant: shards 1/4/5 (rsynced by /tmp/rsync_iq3m.sh or a sibling), a head sidecar
# (ple/head-native-<QUANT>.gguf) and a pack (packs/<QUANT>).
# Patches vs stock Strata: Q5_0 down experts (IQ3_M layers 0-5) and IQ4_XS gate/up experts run on
# CUDA kernels added in src/kernels/cuda/iq_kernels.cu (vec_dot_q5_0_q8_1, vec_dot_iq4_xs_q8_1).
set -uo pipefail

QUANT=${1:?quant name, e.g. IQ3_M}
PORT=${2:?port}
ROOT=/home/bodhi/models/strata
MODEL=$ROOT/model
PY=$ROOT/venv/bin/python
export STRATA_GGUF_PY=$ROOT/third_party/llama.cpp/gguf-py
LOG=$ROOT/logs/finish-$QUANT.log
mkdir -p "$ROOT/logs" "$ROOT/ple" "$ROOT/packs" "$ROOT/mtp"
exec >>"$LOG" 2>&1

LOW=$(echo "$QUANT" | tr 'A-Z' 'a-z')
BASE=Qwen3.8-Flash-Next-heretic-2-$QUANT
XXS=Qwen3.8-Flash-Next-heretic-2-IQ3_XXS
SH1=$MODEL/$BASE-00001-of-00005.gguf
# heretic-2 split: token_embd.weight lives in shard 4, NativeEmbed::load opens only the --native file
NATIVE=$MODEL/$BASE-00004-of-00005.gguf
SH2=$MODEL/$BASE-00002-of-00005.gguf
SH3=$MODEL/$BASE-00003-of-00005.gguf
SH4=$MODEL/$BASE-00004-of-00005.gguf
SH5=$MODEL/$BASE-00005-of-00005.gguf
PLE=$ROOT/ple/ple-iq4nl.gguf
HEAD=$ROOT/ple/head-native-$LOW.gguf
PACK=$ROOT/packs/$LOW
MTPRT=$ROOT/mtp/rt
CTX=${CTX:-131072}
SERVE_CFG=$ROOT/strata-$LOW.json

echo
echo "================ finish_quant $QUANT start $(date -Is) ctx=$CTX port=$PORT ================"

# 1. wait for shards; hardlink the two shared ones from the IQ3_XXS copy (identical LFS oids)
declare -A WANT
OVJSON=$MODEL/layer-experts-override-$QUANT.json   # consolidated-straddle metadata (may grow shard 5)
# exact HF sizes; refresh from the API if a quant is added:
#   IQ3_M:  00001 10945760  00004 47853314784  00005 14813686368
#   IQ4_XS: 00001 10945760  00004 47566147392  00005 22650600960
case "$QUANT" in
  IQ3_M)  WANT[00001]=10945760; WANT[00004]=47853314784; WANT[00005]=14813686368 ;;
  IQ4_XS) WANT[00001]=10945760; WANT[00004]=47566147392; WANT[00005]=22650600960 ;;
  *) echo "[1] unknown quant $QUANT (add its shard sizes)"; exit 1 ;;
esac
for pair in "00002 682434912" "00003 54400261312"; do
  set -- $pair; N=$1; SZ=$2
  F=$MODEL/$BASE-$N-of-00005.gguf
  if [ ! -f "$F" ]; then
    echo "[1] hardlink shared shard $N from the IQ3_XXS copy"
    ln "$MODEL/$XXS-$N-of-00005.gguf" "$F" || { echo "[1] hardlink failed"; exit 1; }
  fi
done
for N in 00001 00004 00005; do
  F=$MODEL/$BASE-$N-of-00005.gguf
  tries=0
  # a consolidated (straddle-fixed) shard may carry appended expert tensors: accept >= the
  # canonical size for shards that consolidate_straddle.py has grown
  until [ -f "$F" ] && [ "$(stat -c%s "$F" 2>/dev/null)" = "${WANT[$N]}" ]; do
    GOT=$(stat -c%s "$F" 2>/dev/null || echo 0)
    if [ "$GOT" -gt "${WANT[$N]}" ] && grep -q "\"shard\"" "$OVJSON" 2>/dev/null; then
      echo "[1] shard $N is $GOT B (consolidated; canonical ${WANT[$N]}) - accepting"
      break
    fi
    tries=$((tries+1)); echo "[1] waiting for shard $N (attempt $tries) $(date -Is)"
    [ "$tries" -gt 2400 ] && { echo "[1] giving up on shard $N"; exit 1; }
    sleep 30
  done
done
# refresh expected sizes from what actually landed (sizes above may be off by header bytes)
echo "[1] all 5 shards present:"; ls -la "$MODEL"/$BASE-*

# 2. PLE + MTP shared artifacts must exist (built by heretic_finish.sh)
if [ ! -s "$PLE" ]; then echo "[2] missing $PLE (built by heretic_finish.sh); aborting"; exit 1; fi
if [ ! -s "$MTPRT/experts.bin" ] || [ ! -s "$MTPRT/draft_vocab.bin" ]; then
  echo "[2] warning: MTP runtime missing; will launch without speculative decoding"
fi

# 3. head sidecar for this quant (arch keys differ per-quant only in name KVs)
if [ ! -s "$HEAD" ]; then
  echo "[3] head sidecar -> $HEAD $(date -Is)"
  "$PY" "$ROOT/tools/make_native_head.py" --arch-shard "$SH1" --tensor-shard "$SH2" --out "$HEAD" || exit 1
else
  echo "[3] head sidecar present"
fi

# 4. pack
if [ ! -s "$OVJSON" ]; then
  "$PY" "$ROOT/tools/consolidate_straddle.py" --model "$SH1" --out "$OVJSON" || exit 1
fi
OV_ARG=""
[ -s "$OVJSON" ] && [ "$(cat "$OVJSON")" != "{}" ] && OV_ARG="--layer-experts-override $OVJSON"
if [ ! -s "$PACK/index.txt" ] || [ ! -s "$PACK/native_experts.txt" ] || [ ! -s "$PACK/tokenizer/vocab.json" ]; then
  echo "[4] pack -> $PACK $(date -Is)"
  rm -rf "$PACK"; mkdir -p "$PACK"
  "$PY" "$ROOT/tools/iq_pack.py" --gguf "$SH1" --out "$PACK" $OV_ARG || exit 1
else
  echo "[4] pack present"
fi
# 4a. heretic-2 stores the canonical-BF16 tensors (hc_*, ssm_alpha/beta, output_hc_*, ple_value) as Q8_0;
#     rewrite those rows as canonical BF16 in dense.bin (idempotent)
"$PY" "$ROOT/tools/repair_heretic_pack.py" --pack "$PACK" --model "$SH1" || exit 1

# 5. config + launcher
CUDA=/home/bodhi/deps/cuda-13.3/opt/cuda/targets/x86_64-linux/lib
MMAP_ARG=', "--mmap-experts"'
PROF_ARG=""
[ -s "$ROOT/data/expert-profile.bin" ] && PROF_ARG=", \"--expert-profile\", \"$ROOT/data/expert-profile.bin\""
MTP_ARG=""; SPEC_ARG=""
# --native points at shard 4 (token_embd), which defeats model_shards()'s literal "-00001-of-"
# sibling discovery - so name every dense shard explicitly, plus the PLE gguf as the engine's auto path does.
DENSE_ARG=""
for N in 00001 00002 00003 00004 00005; do
  DENSE_ARG="$DENSE_ARG, \"--native-dense-gguf\", \"$MODEL/$BASE-$N-of-00005.gguf\""
done
# (ple gguf deliberately NOT in the dense list: it carries general.architecture without the
# full qwen4exp.* keys, and heretic-2 keeps blk.1.ple_key in model shard 4 anyway)
if [ -s "$MTPRT/experts.bin" ] && [ -s "$MTPRT/draft_vocab.bin" ]; then
  MTP_ARG=", \"--mtp\", \"$MTPRT\""
  SPEC_ARG=', "--spec", "4", "--spec-min-p", "0.5"'
fi
ARGS=$(cat <<EOF
["--pack", "$PACK", "--native", "$NATIVE", "--native-head-gguf", "$HEAD", "--ple-gguf", "$PLE"$DENSE_ARG$PROF_ARG,
 "--expert-cache", "auto", "--vram-reserve-mib", "1100"$MMAP_ARG, "--pool-workers", "8", "--prefill", "2048"$SPEC_ARG$MTP_ARG,
 "--max-context", "$CTX", "--kv", "int8"]
EOF
)
cat > "$SERVE_CFG" <<EOF
{
 "exe": "$ROOT/engine/strata",
 "args": $ARGS,
 "cwd": "$ROOT",
 "tokenizer": "$PACK/tokenizer",
 "model_name": "qwen3.8-flash-next-heretic-2-$LOW",
 "log": "$ROOT/logs/strata-$LOW.log",
 "lib_dirs": ["$CUDA"],
 "port": $PORT
}
EOF
cat > "$ROOT/run-$LOW.sh" <<EOF
#!/bin/sh
cd "$ROOT"
exec "$PY" "$ROOT/serve/server.py" --engine strata --config "$SERVE_CFG" --port $PORT
EOF
chmod +x "$ROOT/run-$LOW.sh"
echo "[5] wrote $SERVE_CFG and run-$LOW.sh"

# 6. stop any other strata server (RAM: one engine at a time), then launch
pkill -f 'serve/server.py.*strata-.*\.json' 2>/dev/null && echo "[6] stopped a previous strata server" && sleep 5
echo "[6] launching $QUANT server on :$PORT $(date -Is)"
cd "$ROOT"
nohup "$PY" "$ROOT/serve/server.py" --engine strata --config "$SERVE_CFG" --port $PORT \
    >"$ROOT/logs/server-stdout-$LOW.log" 2>&1 &
echo "[6] server pid $!"

# 7. wait for health, smoke, bench, eval
BENCH=$ROOT/bench/results
mkdir -p "$BENCH"
echo "[7] waiting for :$PORT $(date -Is)"
ready=0
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  sleep 15
done
if [ "$ready" != 1 ]; then echo "[7] endpoint never became ready; see logs/strata-$LOW.log"; tail -40 "$ROOT/logs/strata-$LOW.log" 2>/dev/null; exit 1; fi
echo "[7] ready $(date -Is)"
timeout 900 curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"strata\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}],\"max_tokens\":32,\"reasoning_effort\":\"none\"}" | head -c 800; echo
echo "[7] benchmark $(date -Is)"
"$PY" "$ROOT/bench_endpoint.py" --port "$PORT" --contexts 1024,4096,32768,131072 \
    --max-tokens 192 --effort none --out "$BENCH/${LOW}_$(date +%Y%m%dT%H%M%S).json" || echo "[7] benchmark failed"
echo "[7] thinking benchmark at 4K $(date -Is)"
"$PY" "$ROOT/bench_endpoint.py" --port "$PORT" --contexts 4096 --max-tokens 512 --effort high \
    --out "$BENCH/${LOW}_thinking.json" || true
echo "[7] capability + refusal spot-check $(date -Is)"
"$PY" "$ROOT/eval_heretic.py" --port "$PORT" --out "$BENCH/${LOW}_eval.json" || echo "[7] eval failed"
echo "================ finish_quant $QUANT done $(date -Is) ================"