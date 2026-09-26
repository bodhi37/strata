#!/usr/bin/env bash
# bench_all.sh - the R7 deliverable benchmark.  For each downloaded quant: start the engine on the R7
# config, prove coherence FIRST (a number without coherence is worthless), then measure prefill and
# decode at four context lengths, and write bench/results/r7-<quant>.json + .md.
#
#   bash bench_all.sh            # all three
#   QUANTS="IQ3_XXS" bash bench_all.sh
set -u
ROOT=/home/bodhi/models/strata
PORT=8120
QUANTS=${QUANTS:-"IQ3_XXS IQ3_M IQ4_XS"}
HOT=${HOT:-24}
SPECP=${SPECP:-0.8}
SPECS=${SPECS:-4}
CTXS=${CTXS:-"1024,4096,32768,131072"}
declare -A HEAD=( [IQ3_XXS]="$ROOT/ple/head-native.gguf" [IQ3_M]="$ROOT/ple/head-native-iq3_m.gguf" [IQ4_XS]="$ROOT/ple/head-native-iq4_xs.gguf" )
declare -A PACK=( [IQ3_XXS]="$ROOT/packs/heretic-iq3xxs" [IQ3_M]="$ROOT/packs/iq3_m" [IQ4_XS]="$ROOT/packs/iq4_xs" )
declare -A TOKNAME=( [IQ3_XXS]="iq3xxs" [IQ3_M]="iq3_m" [IQ4_XS]="iq4_xs" )

stop() { pkill -x strata 2>/dev/null; for p in $(pgrep -f 'serve/server\.py'); do kill -9 $p 2>/dev/null; done; sleep 2; }

for Q in $QUANTS; do
  LOG=$ROOT/logs/r7-$Q.log
  CFG=$ROOT/strata-r7-$Q.json
  python3 - "$ROOT" "$Q" "$PORT" "$LOG" "$HOT" "$SPECS" "$SPECP" <<'PY'
import json, sys, os
root, q, port, log, hot, specs, specp = sys.argv[1:8]
nd = lambda p: f"{root}/model/Qwen3.8-Flash-Next-heretic-2-{q}-0000{p}-of-00005.gguf"
pack = {"IQ3_XXS": "heretic-iq3xxs", "IQ3_M": "iq3_m", "IQ4_XS": "iq4_xs"}[q]
head = {"IQ3_XXS": "head-native.gguf", "IQ3_M": "head-native-iq3_m.gguf", "IQ4_XS": "head-native-iq4_xs.gguf"}[q]
args = ["--pack", f"{root}/packs/{pack}", "--native", nd(4),
        "--native-head-gguf", f"{root}/ple/{head}", "--ple-gguf", f"{root}/ple/ple-iq4nl.gguf"]
for p in (1, 2, 3, 4, 5): args += ["--native-dense-gguf", nd(p)]
args += ["--expert-profile", f"{root}/data/profile-r7.bin",
         "--expert-cache", "-2", "--hot-ram-gib", hot, "--mmap-experts",
         "--pool-workers", "12", "--prefill", "2048",
         "--spec", specs, "--spec-min-p", specp, "--mtp", f"{root}/mtp/rt",
         "--max-context", "131072", "--kv", "int8"]
json.dump({"exe": f"{root}/run-engine-mlock.sh", "args": args, "cwd": root,
           "tokenizer": f"{root}/packs/{pack}/tokenizer",
           "model_name": f"qwen3.8-flash-next-heretic-2-{q.lower()}",
           "log": log,
           "lib_dirs": ["/home/bodhi/deps/cuda-13.3/opt/cuda/targets/x86_64-linux/lib"],
           "port": int(port)}, open(f"{root}/strata-r7-{q}.json", "w"), indent=1)
PY
  stop
  rm -f "$LOG"
  cd "$ROOT"
  setsid ./venv/bin/python serve/server.py --engine strata --config "$CFG" --port $PORT > "$LOG.srv" 2>&1 &
  for i in $(seq 1 120); do sleep 3; grep -q "ready:" "$LOG.srv" 2>/dev/null && break; grep -qiE "Traceback|RuntimeError" "$LOG.srv" 2>/dev/null && { echo "[$Q] START FAILED"; tail -5 "$LOG.srv"; continue 2; }; done
  HOTLINE=$(grep -m1 "hot tier" "$LOG")
  echo "===== $Q ====="; echo "  $HOTLINE"

  # ---- 1. COHERENCE GATE: measured before any timing is reported
  ./venv/bin/python eval_heretic.py --port $PORT --max-tokens 96 --out "bench/results/r7-${Q}-eval.json" 2>&1 | tail -2 | sed 's/^/  /'
  COH=$(./venv/bin/python -c "
import json;d=json.load(open('bench/results/r7-$Q-eval.json'))
cap=[r['ok'] for r in d['capability']]
print('PASS' if sum(cap)>=6 else 'FAIL', sum(cap),'/',len(cap))")
  echo "  coherence: $COH"

  # ---- 2. throughput
  ./venv/bin/python bench_endpoint.py --port $PORT --contexts "$CTXS" --max-tokens 192 --out "bench/results/r7-$Q.json" 2>&1 | grep -E "^  prompt|FAILED" | sed 's/^/  /'
  grep -E "experts:|mtp:|ring:|prefetch:" "$LOG" | tail -4 | sed 's/^/  /'
  ./venv/bin/python - "$Q" "$HOTLINE" "$COH" <<'PY'
import json, sys, os
q, hot, coh = sys.argv[1], sys.argv[2], sys.argv[3]
rows = json.load(open(f"bench/results/r7-{q}.json"))
ev = json.load(open(f"bench/results/r7-{q}-eval.json"))
lines = [f"### {q}  (coherence: {coh})", "",
         f"config: `--expert-cache -2 --hot-ram-gib {hot} --mmap-experts --pool-workers 12 --prefill 2048 "
         f"--spec 4 --spec-min-p 0.8 --mtp mtp/rt --kv int8 --expert-profile data/profile-r7.bin`", "",
         "| context | prompt tok | TTFT s | prefill tok/s | decode tok/s |", "| --- | ---: | ---: | ---: | ---: |"]
for r in rows:
    if "error" in r:
        lines.append(f"| {r['target']} | - | - | - | {r['error'][:40]} |")
    else:
        lines.append(f"| ~{r['target']} | {r['prompt_tokens']} | {r['ttft_s']:.1f} | "
                     f"{r['prefill_tok_s']:.0f} | {r['decode_tok_s']:.2f} |")
lines += ["", f"`{hot}`"]
open(f"bench/results/r7-{q}.md", "w").write("\n".join(lines) + "\n")
PY
  cat "bench/results/r7-$Q.md"
  stop
done
echo "DONE"
