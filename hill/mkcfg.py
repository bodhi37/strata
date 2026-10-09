#!/usr/bin/env python3
"""mkcfg.py - derive hillclimb arm configs from the final IQ3_S endpoint config.

Each arm gets its own log path so per-request telemetry stays attributable.
Usage:  python3 hill/mkcfg.py spec8 --set spec=8
        python3 hill/mkcfg.py pool24 --set pool-workers=24 --set log=logs/hc2-pool24.log
Argument names follow the order in args (matches by the value after '--').
"""
import json
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "strata-sc117-iq3s-final.json")

ARG_NAMES = ["pack", "native", "native-head-gguf", "ple-gguf", "native-dense-gguf",
             "expert-profile", "expert-cache", "hot-ram-gib", "mmap-experts", "pool-workers",
             "prefill", "spec", "spec-min-p", "mtp", "max-context", "kv", "suffix-draft",
             "prompt-cache", "expert-cache-per-layer", "vram-reserve-mib", "mtp-max-t",
             "ple-row-cache", "ple-inflight", "short-read"]


def main():
    tag = sys.argv[1]
    sets = {}
    argv = sys.argv[2:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--set":
            k, v = argv[i + 1].split("=", 1)
            sets[k] = v
            i += 2
        elif a.startswith("--set="):
            k, v = a[len("--set="):].split("=", 1)
            sets[k] = v
            i += 1
        else:
            i += 1
    cfg = json.load(open(BASE))
    args = cfg["args"]
    if "log" not in sets:
        sets["log"] = f"logs/hc2-{tag}.log"
    for k, v in sets.items():
        if k == "log":
            cfg["log"] = v
            continue
        if k == "model_name":
            cfg["model_name"] = v
            continue
        if k == "env":
            for ek, ev in [x.split("=", 1) for x in v.split(";") if "=" in x]:
                cfg.setdefault("env", {})[ek] = ev
            continue
        if k not in ARG_NAMES:
            print(f"unknown arg {k}", file=sys.stderr)
            return 2
        i = args.index(f"--{k}") if f"--{k}" in args else None
        val = v
        for cast in (int, float):
            try:
                val = cast(v)
                break
            except ValueError:
                pass
        if i is not None:
            args[i + 1] = val
        else:
            args += [f"--{k}", val]
    out = os.path.join(REPO, f"hill/cfg-{tag}.json")
    json.dump(cfg, open(out, "w"), indent=1)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
