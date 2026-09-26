#!/usr/bin/env bash
# hot_sweep.sh <gib> [extra args...] - start the engine with a <gib> host hot tier from the trace-derived
# profile, run one fixed decode benchmark, print the honest expert-source stats, stop.  One line per run.
set -u
ROOT=/home/bodhi/models/strata
GIB=$1; shift
EXTRA=("$@")
PORT=8110
PACK=${PACK:-heretic-iq3xxs}
MODEL=${MODEL:-IQ3_XXS}
CFG=$ROOT/strata-sweep.json
LOG=$ROOT/logs/sweep-$GIB.log

python3 - "$ROOT" "$GIB" "$PORT" "$PACK" "$MODEL" "$LOG" "${EXTRA[@]}" <<'PY'
import json, sys
root, gib, port, pack, model, log = sys.argv[1:7]
extra = sys.argv[7:]
def nd(p): return f"{root}/model/Qwen3.8-Flash-Next-heretic-2-{model}-0000{p}-of-00005.gguf"
args = ["--pack", f"{root}/packs/{pack}", "--native", nd(4),
        "--native-head-gguf", f"{root}/ple/head-native{'' if model=='IQ3_XXS' else '-'+model.lower().replace('iq3_xxs','')}.gguf",
        "--ple-gguf", f"{root}/ple/ple-iq4nl.gguf"]
for p in (1,2,3,4,5): args += ["--native-dense-gguf", nd(p)]
import os
# head sidecar name differs per quant; pick whichever exists
for cand in [f"{root}/ple/head-native.gguf", f"{root}/ple/head-native-iq3_m.gguf", f"{root}/ple/head-native-iq4_xs.gguf"]:
    if model in cand or (model=="IQ3_XXS" and cand.endswith("head-native.gguf")):
        args[args.index("--native-head-gguf")+1] = cand
        break
args += ["--expert-profile", f"{root}/data/profile-r7.bin",
         "--expert-cache", "-2", "--hot-ram-gib", str(gib), "--mmap-experts",
         "--pool-workers", "12", "--prefill", "2048",
         "--spec", "4", "--spec-min-p", "0.5", "--mtp", f"{root}/mtp/rt",
         "--max-context", "131072", "--kv", "int8"] + extra
json.dump({"exe": f"{root}/run-engine-mlock.sh", "args": args, "cwd": root,
           "tokenizer": f"{root}/packs/{pack}/tokenizer",
           "model_name": f"qwen3.8-flash-next-heretic-2-{model.lower()}",
           "log": log,
           "lib_dirs": ["/home/bodhi/deps/cuda-13.3/opt/cuda/targets/x86_64-linux/lib"],
           "port": int(port)}, open(f"{root}/strata-sweep.json", "w"), indent=1)
PY

pkill -x strata 2>/dev/null
for p in $(pgrep -f 'serve/server\.py'); do kill -9 $p 2>/dev/null; done
sleep 2
rm -f "$LOG"
cd "$ROOT"
setsid ./venv/bin/python serve/server.py --engine strata --config "$CFG" --port $PORT > "$LOG.srv" 2>&1 &
for i in $(seq 1 120); do
  sleep 3
  grep -q "ready:" "$LOG.srv" 2>/dev/null && break
  grep -qiE "Traceback|RuntimeError" "$LOG.srv" 2>/dev/null && { echo "START FAILED"; tail -20 "$LOG.srv"; exit 1; }
done
grep -m1 "hot tier" "$LOG"

./venv/bin/python - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
body = {"model": "m", "messages": [{"role": "user",
        "content": "Explain how a hash table handles collisions, covering chaining and open addressing."}],
        "max_tokens": 150, "reasoning_effort": "none"}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t = time.time()
d = json.load(urllib.request.urlopen(req, timeout=7200))
txt = (d["choices"][0]["message"].get("content") or "")
print("SAMPLE:", repr(txt[:110]))
print("WALL: %.1fs  completion=%d" % (time.time() - t, d["usage"]["completion_tokens"]))
PY
grep -E "experts:|mtp:|prompt tokens|ring:|prefetch:|phases/win" "$LOG" | tail -3
pkill -x strata 2>/dev/null
for p in $(pgrep -f 'serve/server\.py'); do kill -9 $p 2>/dev/null; done
free -g; swapon --show=NAME,USED
