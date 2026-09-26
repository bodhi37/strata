#!/bin/bash
# heretic_finish.sh - bring up Qwen3.8-Flash-Next Heretic-2 (abliterated) IQ3_XXS under Strata.
#
# Idempotent: every step skips work whose output already exists.  Started under nohup so it
# survives the agent session.  Log: logs/finish.log
#
# What makes this build different from a stock Strata install:
#   * the community "heretic-2" GGUF pins the 28.8 GB n-gram (PLE) table to Q8_0 (54.4 GB);
#     Strata's ngram kernel only accepts IQ4_NL (90 B/row), so tools/make_ple_iq4nl.py rewrites
#     it -> ple/ple-iq4nl.gguf (28.8 GB).
#   * the model is a 5-shard split (00001..00005).  tools/iq_pack.py and the engine both
#     auto-discover siblings, but --native must be pointed at a shard that holds token_embd.weight
#     (heretic-2 split: 00001 is an empty guard, token_embd lives in 00004)
#     (shard 4), not the metadata-only shard 1.
set -uo pipefail

ROOT=/home/bodhi/models/strata
MODEL=$ROOT/model
SRC=x64:/mnt/1tbssd/ssh-downloads/aimodels/qwen38-fn-heretic2-iq3xxs
PY=$ROOT/venv/bin/python
export STRATA_GGUF_PY=$ROOT/third_party/llama.cpp/gguf-py
LOG=$ROOT/logs/finish.log
mkdir -p "$ROOT/logs" "$ROOT/ple" "$ROOT/packs" "$ROOT/mtp"
exec >>"$LOG" 2>&1

BASE=Qwen3.8-Flash-Next-heretic-2-IQ3_XXS
SH1=$MODEL/$BASE-00001-of-00005.gguf
# heretic-2 split: --native must name a shard containing token_embd.weight (NativeEmbed::load
# opens exactly this one file); shard 1 is an empty guard, token_embd.weight is in shard 4.
NATIVE=$MODEL/$BASE-00004-of-00005.gguf
SH2=$MODEL/$BASE-00002-of-00005.gguf
SH3=$MODEL/$BASE-00003-of-00005.gguf
SH4=$MODEL/$BASE-00004-of-00005.gguf
SH5=$MODEL/$BASE-00005-of-00005.gguf
PLE=$ROOT/ple/ple-iq4nl.gguf
HEAD=$ROOT/ple/head-native.gguf
PACK=$ROOT/packs/heretic-iq3xxs
MTP=$ROOT/mtp
MTPRT=$MTP/rt
CTX=${CTX:-131072}
PORT=${PORT:-8101}

echo
echo "================ heretic_finish start $(date -Is)  ctx=$CTX ================"

# ---------------------------------------------------------------- 1. downloads
echo "[1] waiting for any in-flight rsync to exit ..."
while pgrep -f 'rsync.*qwen38-fn-heretic2-iq3xxs' >/dev/null; do sleep 30; done

want() { case "$1" in 00003) echo 54400261312;; 00004) echo 47528931424;; esac; }
for n in 00003 00004; do
  f=$MODEL/$BASE-$n-of-00005.gguf
  w=$(want "$n")
  tries=0
  until [ -f "$f" ] && [ "$(stat -c%s "$f" 2>/dev/null)" = "$w" ]; do
    tries=$((tries+1))
    echo "[1] rsync shard $n (attempt $tries) $(date -Is)"
    rsync -a --partial --no-inc-recursive "$SRC/$BASE-$n-of-00005.gguf" "$MODEL/" || sleep 15
    [ "$tries" -gt 200 ] && { echo "[1] giving up on shard $n"; exit 1; }
  done
  echo "[1] shard $n complete ($w bytes)"
done
ls -la "$MODEL"

# ---------------------------------------------------------------- 2. PLE requant (Q8_0 -> IQ4_NL)
if [ ! -s "$PLE" ]; then
  echo "[2] requantising PLE table -> $PLE $(date -Is)"
  "$PY" "$ROOT/tools/make_ple_iq4nl.py" --src "$SH3" --out "$PLE" --verify 1024 || exit 1
else
  echo "[2] PLE already present: $(stat -c%s "$PLE") bytes"
fi

# ---------------------------------------------------------------- 3. pack (native experts + dense)
if [ ! -s "$PACK/index.txt" ] || [ ! -s "$PACK/native_experts.txt" ] || [ ! -s "$PACK/tokenizer/vocab.json" ]; then
  echo "[3] building pack -> $PACK $(date -Is)"
  rm -rf "$PACK"; mkdir -p "$PACK"
  OV=$MODEL/layer-experts-override-IQ3_XXS.json
  [ -s "$OV" ] || "$PY" "$ROOT/tools/consolidate_straddle.py" --model "$SH1" --out "$OV" || exit 1
  OV_ARG=""
  [ -s "$OV" ] && [ "$(cat "$OV")" != "{}" ] && OV_ARG="--layer-experts-override $OV"
  "$PY" "$ROOT/tools/iq_pack.py" --gguf "$SH1" --out "$PACK" $OV_ARG || exit 1
else
  echo "[3] pack already present"
fi
# 3a. heretic-2 stores the canonical-BF16 tensors (hc_*, ssm_alpha/beta, output_hc_*, ple_value) as Q8_0;
#     rewrite those rows as canonical BF16 in dense.bin (idempotent)
"$PY" "$ROOT/tools/repair_heretic_pack.py" --pack "$PACK" --model "$SH1" || exit 1

# ---------------------------------------------------------------- 3b. native head sidecar
# `--native` auto-sets --native-head-gguf to the shard it is given, but NativeHead::load opens
# ONE file and checks the architecture guard AND output.weight in it.  Here the guard keys live in
# shard 1 (0 tensors) and output.weight in shard 2, so build a one-tensor sidecar.
if [ ! -s "$HEAD" ]; then
  echo "[3b] building native head sidecar -> $HEAD $(date -Is)"
  "$PY" "$ROOT/tools/make_native_head.py" --arch-shard "$SH1" --tensor-shard "$SH2" --out "$HEAD" || exit 1
else
  echo "[3b] native head sidecar present: $(stat -c%s "$HEAD") bytes"
fi

# ---------------------------------------------------------------- 4. MTP draft layer
if [ ! -s "$MTPRT/experts.bin" ] || [ ! -s "$MTPRT/draft_vocab.bin" ]; then
  echo "[4] MTP draft layer $(date -Is)"
  mkdir -p "$MTPRT"
  [ -s "$MTP/mtp-manifest.json" ] || "$PY" "$ROOT/tools/mtp_fetch.py" fetch --out "$MTP" || echo "[4] mtp_fetch failed (continuing without MTP)"
  if [ -s "$MTP/mtp-manifest.json" ] && [ ! -s "$MTP/mtp-q2_0.gguf" ]; then
    "$PY" "$ROOT/tools/mtp_pack.py" --src "$MTP" --experts q2_0 --out "$MTP/mtp-q2_0.gguf" || true
  fi
  if [ -s "$MTP/mtp-q2_0.gguf" ]; then
    "$PY" "$ROOT/tools/mtp_rt.py" --gguf "$MTP/mtp-q2_0.gguf" --out "$MTPRT" || true
  fi
  cp -f "$ROOT/data/draft_vocab.bin" "$MTPRT/draft_vocab.bin"
else
  echo "[4] MTP runtime already present"
fi

# ---------------------------------------------------------------- 5. config + run script
TOK=$PACK/tokenizer
CUDA=/home/bodhi/deps/cuda-13.3/opt/cuda/targets/x86_64-linux/lib
# This host has ~25 GiB usable RAM for a ~58 GB expert set, so the pinned resident arena cannot fit:
# opt into the MapViewOfFile / OS-page-cache path.  Override with STRATA_MMAP_EXPERTS=0 to test the arena.
MMAP_ARG=""
[ "${STRATA_MMAP_EXPERTS:-1}" = "1" ] && MMAP_ARG=', "--mmap-experts"'
PROF_ARG=""
[ -s "$ROOT/data/expert-profile.bin" ] && PROF_ARG=", \"--expert-profile\", \"$ROOT/data/expert-profile.bin\""
MTP_ARG=""
SPEC_ARG=""
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
else
  echo "[5] MTP runtime unavailable; launching without speculative decoding"
fi
ARGS=$(cat <<EOF
["--pack", "$PACK", "--native", "$NATIVE", "--native-head-gguf", "$HEAD", "--ple-gguf", "$PLE"$DENSE_ARG$PROF_ARG,
 "--expert-cache", "auto", "--vram-reserve-mib", "1100"$MMAP_ARG, "--pool-workers", "8", "--prefill", "2048"$SPEC_ARG$MTP_ARG,
 "--max-context", "$CTX", "--kv", "int8"]
EOF
)
CFG=$ROOT/strata-heretic.json
cat > "$CFG" <<EOF
{
 "exe": "$ROOT/engine/strata",
 "args": $ARGS,
 "cwd": "$ROOT",
 "tokenizer": "$TOK",
 "model_name": "qwen3.8-flash-next-heretic-2-iq3xxs",
 "log": "$ROOT/strata-heretic.log",
 "lib_dirs": ["$CUDA"],
 "port": $PORT
}
EOF
cat > "$ROOT/run-heretic.sh" <<EOF
#!/bin/sh
cd "$ROOT"
exec "$PY" "$ROOT/serve/server.py" --engine strata --config "$CFG" --port $PORT
EOF
chmod +x "$ROOT/run-heretic.sh"
echo "[5] wrote $CFG and run-heretic.sh"

# ---------------------------------------------------------------- 6. launch
if pgrep -f 'serve/server.py.*strata-heretic.json' >/dev/null; then
  echo "[6] server already running"
else
  echo "[6] launching server $(date -Is)"
  cd "$ROOT"
  nohup "$PY" "$ROOT/serve/server.py" --engine strata --config "$CFG" --port $PORT \
      >"$ROOT/logs/server-stdout.log" 2>&1 &
  echo "[6] server pid $!"
fi
echo "================ heretic_finish done $(date -Is) ================"

# ---------------------------------------------------------------- 7. smoke test + benchmark
BENCH=$ROOT/bench/results
mkdir -p "$BENCH"
echo "[7] waiting for the endpoint on :$PORT (model load can take minutes) $(date -Is)"
ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 15
done
if [ "$ready" != 1 ]; then
  echo "[7] endpoint did not become ready; see $ROOT/strata-heretic.log"; tail -40 "$ROOT/strata-heretic.log" 2>/dev/null; exit 1
fi
echo "[7] endpoint ready $(date -Is)"
curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 400; echo
echo "[7] smoke test (short, thinking high)"
timeout 900 curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":32,"reasoning_effort":"none"}' \
  | head -c 1200; echo
echo "[7] benchmark $(date -Is)"
"$PY" "$ROOT/bench_endpoint.py" --port "$PORT" --contexts 1024,4096,32768,131072 \
    --max-tokens 192 --effort none --out "$BENCH/heretic_iq3xxs_$(date +%Y%m%dT%H%M%S).json" \
    || echo "[7] benchmark failed"
echo "[7] also measure a thinking request at 4K"
"$PY" "$ROOT/bench_endpoint.py" --port "$PORT" --contexts 4096 --max-tokens 512 --effort high \
    --out "$BENCH/heretic_iq3xxs_thinking.json" || echo "[7] thinking benchmark failed"
echo "[7] capability + refusal spot-check $(date -Is)"
"$PY" "$ROOT/eval_heretic.py" --port "$PORT" --out "$BENCH/heretic_iq3xxs_eval.json" || echo "[7] eval failed"
echo "================ heretic_finish all done $(date -Is) ================"
