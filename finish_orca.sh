#!/bin/bash
# finish_orca.sh - port OrcaRouter Qwen3.8-Flash-Next-Uncensored IQ4_XS (3-shard) to Strata.
#   bash finish_orca.sh [PORT]   # default 8104 (8123 stays = IQ3_XXS, do not touch)
#
# Pipeline (mirrors heretic_finish.sh / finish_quant.sh, adapted for 3-shard split):
#   1. wait for pull_orca.sh shards in model/
#   2. probe shards -> find arch/token_embd/output/PLE layout
#   3. head sidecar (ple/head-native-orca-iq4xs.gguf)
#   4. pack (packs/orca-iq4xs) via iq_pack.py + repair_heretic_pack.py (best-effort)
#   5. PLE: reuse ple-iq4nl.gguf if Orca has no PLE table, else requant to ple-orca-iq4nl.gguf
#   6. config strata-orca-iq4xs.json (clone of r23-wrapup knobs) + bench
# Leaves strata-r23-wrapup.json / port 8123 (IQ3_XXS) untouched.
set -uo pipefail
ROOT=/home/bodhi/models/strata
MODEL=$ROOT/model
DEST=/home/bodhi/models/qwen3.8-flash-next-orcarouter-uncensored-iq4xs
PY=$ROOT/venv/bin/python
export STRATA_GGUF_PY=$ROOT/third_party/llama.cpp/gguf-py
PORT=${1:-8104}
CTX=${CTX:-131072}
LOG=$ROOT/logs/finish-orca.log
mkdir -p "$ROOT/logs" "$ROOT/ple" "$ROOT/packs" "$ROOT/mtp"
exec >>"$LOG" 2>&1

BASE=Qwen3.8-Flash-Next-Uncensored-IQ4_XS
SH1=$MODEL/$BASE-00001-of-00003.gguf
SH2=$MODEL/$BASE-00002-of-00003.gguf
SH3=$MODEL/$BASE-00003-of-00003.gguf
PLE_SHARED=$ROOT/ple/ple-iq4nl.gguf
PLE_ORCA=$ROOT/ple/ple-orca-iq4nl.gguf
HEAD=$ROOT/ple/head-native-orca-iq4xs.gguf
PACK=$ROOT/packs/orca-iq4xs
MTPRT=$ROOT/mtp/rt
CFG=$ROOT/strata-orca-iq4xs.json

echo
echo "================ finish_orca start $(date -Is) ctx=$CTX port=$PORT ================"

# 1. shards (pull_orca.sh hardlinks them into model/)
for N in 00001 00002 00003; do
  F=$MODEL/$BASE-$N-of-00003.gguf
  tries=0
  until [ -s "$F" ]; do
    # also accept them in DEST/ and link over
    if [ -s "$DEST/$BASE-$N-of-00003.gguf" ]; then
      ln -f "$DEST/$BASE-$N-of-00003.gguf" "$F" 2>/dev/null || cp -f "$DEST/$BASE-$N-of-00003.gguf" "$F"
    fi
    [ -s "$F" ] && break
    tries=$((tries+1)); echo "[1] waiting for shard $N (attempt $tries) $(date -Is)"
    [ "$tries" -gt 1440 ] && { echo "[1] giving up on shard $N"; exit 1; }
    sleep 60
  done
done
ls -lh "$MODEL"/$BASE-*
[ -s "$DEST/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf" ] || echo "[1] note: mmproj not yet pulled (vision optional)"

# 2. probe layout
echo "[2] probing shards $(date -Is)"
"$PY" "$ROOT/tools/probe_gguf.py" "$SH1" 2>&1 | head -n 40 || true
for S in "$SH1" "$SH2" "$SH3"; do
  echo "--- $S"
  "$PY" -c "
import sys; sys.path.insert(0,'$ROOT/tools')
from gguf_reader import GGUFFile
g=GGUFFile('$S')
names={t.name for t in g.tensors}
for k in ['token_embd.weight','output.weight','blk.1.ple_key','blk.0.ffn_gate_inp.weight','output_norm.weight']:
  print(' ',k, 'YES' if k in names else 'no')
print('  ntensors', len(names))
print('  arch', g.metadata.get('general.architecture'), g.metadata.get('qwen4exp.block_count'))
" 2>&1 | head -n 15 || true
done

# find tensor shard for output.weight + arch shard for guard keys
ARCH_SHARD=$SH1
TENSOR_SHARD=$SH2
for S in "$SH1" "$SH2" "$SH3"; do
  HAS_OUT=$("$PY" -c "
import sys; sys.path.insert(0,'$ROOT/tools')
from gguf_reader import GGUFFile
g=GGUFFile('$S'); print('yes' if 'output.weight' in {t.name for t in g.tensors} else 'no')" 2>/dev/null | tail -1)
  [ "$HAS_OUT" = "yes" ] && TENSOR_SHARD=$S
done
echo "[2] arch=$ARCH_SHARD tensor=$TENSOR_SHARD"

# 3. head sidecar
if [ ! -s "$HEAD" ]; then
  echo "[3] head sidecar -> $HEAD $(date -Is)"
  "$PY" "$ROOT/tools/make_native_head.py" --arch-shard "$ARCH_SHARD" --tensor-shard "$TENSOR_SHARD" --out "$HEAD" || exit 1
else
  echo "[3] head sidecar present"
fi

# 4. pack
if [ ! -s "$PACK/index.txt" ] || [ ! -s "$PACK/native_experts.txt" ] || [ ! -s "$PACK/tokenizer/vocab.json" ]; then
  echo "[4] pack -> $PACK $(date -Is)"
  OV=$MODEL/layer-experts-override-orca-iq4xs.json
  [ -s "$OV" ] || "$PY" "$ROOT/tools/consolidate_straddle.py" --model "$SH1" --out "$OV" || echo "[4] consolidate_straddle failed (continuing without override)"
  OV_ARG=""
  [ -s "$OV" ] && [ "$(cat "$OV")" != "{}" ] && OV_ARG="--layer-experts-override $OV"
  rm -rf "$PACK"; mkdir -p "$PACK"
  # shellcheck disable=SC2086
  "$PY" "$ROOT/tools/iq_pack.py" --gguf "$SH1" --out "$PACK" $OV_ARG || exit 1
else
  echo "[4] pack present"
fi
"$PY" "$ROOT/tools/repair_heretic_pack.py" --pack "$PACK" --model "$SH1" || echo "[4a] repair_heretic_pack skipped/failed (non-fatal for Orca)"

# 5. PLE: check if Orca shards carry a PLE table; if so requant, else reuse shared
HAS_PLE=$("$PY" -c "
import sys; sys.path.insert(0,'$ROOT/tools')
from gguf_reader import GGUFFile
found=False
for s in ['$SH1','$SH2','$SH3']:
  try:
    g=GGUFFile(s)
    if any('ple' in t.name.lower() for t in g.tensors):
      found=True; break
  except Exception: pass
print('yes' if found else 'no')" 2>/dev/null | tail -1)
echo "[5] orca carries PLE table: $HAS_PLE"
PLE=$PLE_SHARED
if [ "$HAS_PLE" = "yes" ]; then
  if [ ! -s "$PLE_ORCA" ]; then
    echo "[5] requantising Orca PLE -> $PLE_ORCA $(date -Is)"
# Orca split: PLE table (blk.1.ple_*) lives in shard 1 (heretic kept it in shard 3).
    PLE_SRC=$SH1
    "$PY" "$ROOT/tools/make_ple_iq4nl.py" --src "$PLE_SRC" --out "$PLE_ORCA" --verify 1024 || { echo "[5] orca PLE requant failed, falling back to shared"; PLE=$PLE_SHARED; }
    [ -s "$PLE_ORCA" ] && PLE=$PLE_ORCA
  else
    PLE=$PLE_ORCA
  fi
fi
[ -s "$PLE" ] || { echo "[5] missing PLE $PLE; aborting"; exit 1; }
echo "[5] using PLE $PLE ($(stat -c%s "$PLE") bytes)"

# 6. config (clone r23-wrapup knobs; IQ3_XXS endpoint on 8123 untouched)
TOK=$PACK/tokenizer
CUDA=/home/bodhi/deps/cuda-13.3/opt/cuda/targets/x86_64-linux/lib
NATIVE=$TENSOR_SHARD
# --native must name a shard containing token_embd.weight; probe for it
for S in "$SH1" "$SH2" "$SH3"; do
  HAS_EMB=$("$PY" -c "
import sys; sys.path.insert(0,'$ROOT/tools')
from gguf_reader import GGUFFile
g=GGUFFile('$S'); print('yes' if 'token_embd.weight' in {t.name for t in g.tensors} else 'no')" 2>/dev/null | tail -1)
  [ "$HAS_EMB" = "yes" ] && NATIVE=$S
done
echo "[6] native=$NATIVE"
DENSE_ARG=""
for N in 00001 00002 00003; do
  DENSE_ARG="$DENSE_ARG, \"--native-dense-gguf\", \"$MODEL/$BASE-$N-of-00003.gguf\""
done
PROF_ARG=""
[ -s "$ROOT/data/profile-r14.bin" ] && PROF_ARG=", \"--expert-profile\", \"$ROOT/data/profile-r14.bin\""
MTP_ARG=""; SPEC_ARG=""
if [ -s "$MTPRT/experts.bin" ] && [ -s "$MTPRT/draft_vocab.bin" ]; then
  MTP_ARG=", \"--mtp\", \"$MTPRT\""
  SPEC_ARG=', "--spec", "4", "--spec-min-p", "0.8"'
fi
ARGS=$(cat <<EOF
["--pack", "$PACK", "--native", "$NATIVE", "--native-head-gguf", "$HEAD", "--ple-gguf", "$PLE"$DENSE_ARG$PROF_ARG,
 "--expert-cache", "1500", "--hot-ram-gib", "24.0", "--mmap-experts", "--pool-workers", "20", "--prefill", "16384"$SPEC_ARG$MTP_ARG,
 "--max-context", "$CTX", "--kv", "q4_0", "--suffix-draft", "3", "--prompt-cache", "2"]
EOF
)
cat > "$CFG" <<EOF
{
 "exe": "$ROOT/run-engine-mlock.sh",
 "args": $ARGS,
 "cwd": "$ROOT",
 "tokenizer": "$TOK",
 "model_name": "qwen3.8-flash-next-orca-iq4xs",
 "log": "$ROOT/logs/strata-orca-iq4xs.log",
 "lib_dirs": ["$CUDA"],
 "port": $PORT,
 "env": {
  "STRATA_RSPLIT": "1"
 }
}
EOF
echo "[6] wrote $CFG"
cat "$CFG"

# 7. profile refresh (routing counts for the new quant; best-effort)
echo "[7] profile note: data/profile-r14.bin is heretic-derived; rebuild from Orca traces after first run (see tools/make_profile_from_counts.py)"

echo "FINISH_ORCA_DONE $(date -Is)"
echo "next: bash srv.sh strata-orca-iq4xs.json $PORT"
