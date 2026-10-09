#!/usr/bin/env python3
"""scansweep.py - launch sc117 server with config overrides, bench, kill. One line per run.

Usage: scansweep.py LABEL [--arg K V ...] [--env K=V ...] [--ctxs 1024,8192] [--ntok 128]
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request

ROOT = "/home/bodhi/models/strata"
BASE = {
 "exe": f"{ROOT}/run-engine-mlock.sh",
 "args": [
  "--pack", f"{ROOT}/packs/sc117-iq3s",
  "--native", "/home/bodhi/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
  "--native-head-gguf", f"{ROOT}/ple/head-native-sc117.gguf",
  "--ple-gguf", "/home/bodhi/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf",
  "--native-dense-gguf", "/home/bodhi/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
  "--native-dense-gguf", "/home/bodhi/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf",
  "--expert-profile", f"{ROOT}/data/profile-sc117-r2.bin",
  "--expert-cache", "800",
  "--hot-ram-gib", "24.5",
  "--mmap-experts",
  "--pool-workers", "16",
  "--prefill", "8192",
  "--spec", "4",
  "--spec-min-p", "0.8",
  "--mtp", "/home/bodhi/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/strata/rt",
  "--max-context", "131072",
  "--kv", "int8",
  "--suffix-draft", "3",
  "--prompt-cache", "2",
 ],
 "cwd": ROOT,
 "tokenizer": f"{ROOT}/packs/sc117-iq3s/tokenizer",
 "model_name": "qwen3.8-flash-next-gsq-rco-abliterated-iq3s",
 "lib_dirs": ["/home/bodhi/.deps/cuda-13.3/opt/cuda/targets/x86_64-linux/lib"],
 "env": {"STRATA_PREFILL_TILE": "768", "STRATA_RSPLIT": "1", "STRATA_PREFILL_ADMIT": "0", "STRATA_STATIC_TIER": "1"},
}

def set_arg(args, key, val):
    if key in args:
        args[args.index(key) + 1] = str(val)
    else:
        args += [key, str(val)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--arg", nargs=2, action="append", default=[], metavar=("KEY", "VAL"))
    ap.add_argument("--flag", action="append", default=[])
    ap.add_argument("--noflag", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--unenv", action="append", default=[])
    ap.add_argument("--ctxs", default="1024,8192")
    ap.add_argument("--ntok", default="128")
    ap.add_argument("--port", default="8124")
    a = ap.parse_args()

    cfg = json.loads(json.dumps(BASE))
    for k, v in a.arg:
        set_arg(cfg["args"], k, v)
    for f in a.flag:
        if f not in cfg["args"]: cfg["args"].append(f)
    for f in a.noflag:
        if f in cfg["args"]: cfg["args"].remove(f)
    for e in a.env:
        k, _, v = e.partition("=")
        cfg["env"][k] = v
    for k in a.unenv:
        cfg["env"].pop(k, None)
    cfg["port"] = int(a.port)
    log = f"{ROOT}/logs/scan-{a.label}.log"
    cfg["log"] = log
    cfgp = f"{ROOT}/scan-{a.label}.json"
    open(cfgp, "w").write(json.dumps(cfg, indent=1))

    subprocess.run(["bash", "-c", "pkill -f 'serve/server\\.py' 2>/dev/null; pkill -x strata 2>/dev/null; sleep 3"], check=False)
    env = dict(os.environ)
    env.update(cfg["env"])
    fp = open(f"{ROOT}/logs/scan-{a.label}.srv", "w")
    p = subprocess.Popen([f"{ROOT}/venv/bin/python", f"{ROOT}/serve/server.py", "--engine", "strata",
                          "--config", cfgp, "--port", a.port], cwd=ROOT, env=env, stdout=fp, stderr=subprocess.STDOUT)
    ready = None
    for i in range(180):
        time.sleep(3)
        if p.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=3) as r:
                if b"qwen" in r.read(): ready = True; break
        except Exception: pass
    out = {"label": a.label, "ready": bool(ready)}
    if ready:
        try:
            b = subprocess.run([f"{ROOT}/venv/bin/python", f"{ROOT}/bench_endpoint.py", "--port", a.port,
                                "--contexts", a.ctxs, "--max-tokens", a.ntok],
                               cwd=ROOT, capture_output=True, text=True, timeout=1800)
            try:
                res = json.loads(b.stdout[b.stdout.index("["):])
                out["results"] = [{k: round(r[k], 2) for k in ("prompt_tokens", "prefill_tok_s", "decode_tok_s", "ttft_s")} for r in res]
            except Exception:
                out["error"] = b.stdout[-500:] + b.stderr[-500:]
        except Exception as ex: out["error"] = str(ex)
        try:
            out["stats"] = [l.strip() for l in open(log).read().splitlines() if "tokens/s" in l or "miss" in l][-3:]
        except Exception: pass
    else:
        try: out["tail"] = open(log).read()[-400:] if os.path.exists(log) else "no log"
        except Exception: pass
    p.terminate(); time.sleep(3)
    subprocess.run(["bash", "-c", "pkill -f 'serve/server\\.py' 2>/dev/null; pkill -x strata 2>/dev/null"], check=False)
    print(json.dumps(out))
    open(f"{ROOT}/logs/scan-{a.label}.json", "w").write(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
