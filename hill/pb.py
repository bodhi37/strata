#!/usr/bin/env python3
"""pb.py - prefill bench: long real-text prompts (source files) at a few sizes, max_tokens 1, unique nonce per request
so no prompt cache can reuse anything.  The engine's own "prompt N tokens = R reused + K read in M ms" line is the
measure (TTFT and HTTP never leak in).

  hill/pb.py --tag u6 --cfg hill/cfg-u6.json [--no-restart] [--sizes 4000,16000,32000] [--reps 1]
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hill"))
from hc import KEY  # noqa: E402

PP = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")
SRC = "/home/user/strata-up/src"   # a fixed corpus (upstream's sources), never the engine's own working tree


def corpus() -> str:
    parts = []
    for root, _, files in sorted(os.walk(SRC)):
        for f in sorted(files):
            if f.endswith((".cpp", ".cu", ".hpp")):
                try:
                    parts.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
                except OSError:
                    pass
    return "\n".join(parts)


def last_pp(log: str, after: int):
    with open(log, "rb") as f:
        f.seek(after)
        txt = f.read().decode("utf-8", "replace")
    m = PP.findall(txt)
    return m[-1] if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--port", type=int, default=8126)
    ap.add_argument("--sizes", default="4000,16000,32000")   # approximate tokens (3.2 chars/token for code)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--no-restart", action="store_true")
    a = ap.parse_args()
    cfgp = a.cfg if os.path.isabs(a.cfg) else os.path.join(REPO, a.cfg)
    cfg = json.load(open(cfgp))
    log = cfg["log"] if os.path.isabs(cfg["log"]) else os.path.join(cfg.get("cwd", REPO), cfg["log"])
    host = os.environ.get("STRATA_HOST", "127.0.0.1")
    srv = os.environ.get("STRATA_SRV", os.path.join(REPO, "srv.sh"))
    if not a.no_restart:
        subprocess.run(["bash", srv, "stop"], capture_output=True, timeout=120)
        r = subprocess.run(["bash", srv, cfgp, str(a.port)], capture_output=True, text=True, timeout=900)
        print(r.stdout.strip(), flush=True)
        if "UP after" not in r.stdout:
            return 1
    text = corpus()
    rng = random.Random(4321)
    rows = []
    for rep in range(a.reps):
        for n in [int(x) for x in a.sizes.split(",")]:
            chars = int(n * 3.2)
            start = rng.randrange(0, max(1, len(text) - chars))
            nonce = f"[session {time.time_ns()} {rng.random():.9f}]\n"
            prompt = nonce + "Summarize what this code does:\n\n" + text[start:start + chars]
            body = {"model": cfg.get("model_name", "m"), "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1, "stream": False, "reasoning_effort": "none", "temperature": 1.0}
            at = os.path.getsize(log)
            req = urllib.request.Request(f"http://{host}:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            t0 = time.time()
            urllib.request.urlopen(req, timeout=3600).read()
            wall = time.time() - t0
            time.sleep(0.5)
            m = last_pp(log, at)
            if not m:
                print(f"  {n}: no prompt line in the log (wall {wall:.1f} s)", flush=True)
                continue
            tok, reused, read, ms, tps = int(m[0]), int(m[1]), int(m[2]), int(m[3]), float(m[4])
            rows.append(dict(size=n, tokens=tok, reused=reused, read=read, ms=ms, tps=tps, wall=wall))
            print(f"  {a.tag} {tok:6d} tok (reused {reused}): {read} read in {ms} ms = {tps:.0f} tok/s (wall {wall:.1f} s)",
                  flush=True)
    tot_read = sum(r["read"] for r in rows); tot_ms = sum(r["ms"] for r in rows)
    agg = tot_read / (tot_ms / 1000) if tot_ms else 0
    print(f"[pb] {a.tag}: AGG prefill {agg:.0f} tok/s over {tot_read} tokens", flush=True)
    os.makedirs(os.path.join(REPO, "bench", "results", "pb"), exist_ok=True)
    with open(os.path.join(REPO, "bench", "results", "pb", f"{a.tag}.json"), "w") as f:
        json.dump({"tag": a.tag, "cfg": a.cfg, "agg": agg, "rows": rows}, f, indent=1)
    with open(os.path.join(REPO, "bench", "results", "pb", "LEDGER.md"), "a") as f:
        f.write(f"| {time.strftime('%m-%d %H:%M')} | {a.tag} | {agg:.0f} | {tot_read} | {a.cfg} | "
                + " ".join(f"{r['tokens']}:{r['tps']:.0f}" for r in rows) + " |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
