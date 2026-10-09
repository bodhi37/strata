#!/usr/bin/env python3
"""hc.py - the R26 hillclimb driver.

One arm = one config + one restart + n bench samples per context size, then the
per-request engine telemetry (experts/ring/mtp/phases lines) is parsed out of the
engine log so every arm lands as one comparable markdown row.

  hill/hc.py --tag base --cfg strata-sc117-iq3s-final.json --port 8126 \
             --ctxs 4096 --n 3
  hill/hc.py --tag spec8 --cfg hill/cfg-spec8.json --port 8126 --ctxs 4096,26214 --n 3

Rules kept from REPORT.md R22: n>=3 samples per arm (single samples lie by 3x on
this box), warm discipline is identical per arm (fresh prefill each sample, the
prompt-cache is cleared by the restart), and the engine telemetry is read from the
config's own log path so what is compared is what the engine measured.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import urllib.request
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = ""
try:
    with open(os.path.expanduser("~/.config/qwen-serve/api-key")) as kf:
        KEY = kf.read().strip()
except OSError:
    KEY = "local"


def sh(cmd: str, timeout: int = 900) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout


def wait_ready(host: str, port: int, model_hint: str = "qwen", timeout_s: int = 420) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(f"http://{host}:{port}/v1/models",
                                         headers={"Authorization": f"Bearer {KEY}"})
            d = urllib.request.urlopen(req, timeout=3).read().decode()
            if model_hint in d:
                return True
        except Exception:
            pass
        if "SERVER DIED during startup" in sh("tmux ls 2>/dev/null; true"):
            pass
        time.sleep(4)
    return False


def one_chat(host: str, port: int, model: str, prompt: str, max_tokens: int,
             effort: str = "none", timeout: float = 7200):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True, "reasoning_effort": effort}
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                                  data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json",
                                           "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    tfirst = None
    n = 0
    usage = None
    first_txt = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                if tfirst is None:
                    tfirst = time.time()
                    first_txt = (delta.get("content") or delta.get("reasoning_content") or "")[:80]
                n += 1
    t1 = time.time()
    return t0, tfirst, t1, n, usage, first_txt


def bench_ctx(host, port, model, target, max_tokens, n, log_path=None) -> list[dict]:
    # same prompt generator as bench_endpoint.py (the documented harness)
    BASE = ("def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n"
            "The quick brown fox jumps over the lazy dog.  A distributed system reaches consensus\n"
            "when a quorum of replicas agrees on a value despite crashes and message loss; the\n"
            "safety property must hold across every possible interleaving of events.\n\n")
    approx = max(64, target) * 4
    body = (BASE * (approx // len(BASE) + 1))[:approx]
    prompt = ("Read the material below.  Then, in your own words, explain what it says and why it matters, "
              "at length, covering every point in order.  Do not summarise briefly; be thorough.\n\n" + body)
    rows = []
    for i in range(n):
        t0, tf, t1, ntok, usage, first = one_chat(host, port, model, prompt, max_tokens)
        pt = (usage or {}).get("prompt_tokens")
        ct = (usage or {}).get("completion_tokens", ntok)
        ttft = (tf - t0) if tf else float("nan")
        dec = (t1 - tf) if tf else float("nan")
        # per-sample telemetry: the engine logs one block per request; a REUSED-prompt
        # sample's block has no prefill in it, so its ringwait/phases are decode-only.
        tele = engine_telemetry(log_path, want_prompt=pt, want_gen=ct) if log_path else {}
        row = {"i": i, "prompt_tokens": pt, "completion_tokens": ct, "ttft_s": ttft,
               "prefill_tok_s": (pt / ttft) if (pt and ttft and ttft > 0) else float("nan"),
               "decode_tok_s": ((ct - 1) / dec) if (dec and dec > 0) else float("nan"),
               "wall_s": t1 - t0, "sample": first, "tele": tele}
        rows.append(row)
        print(f"    sample {i}: ptok={pt} ctok={ct} ttft={ttft:.1f}s "
              f"prefill={row['prefill_tok_s']:.0f} decode={row['decode_tok_s']:.2f} "
              f"wall={t1-t0:.1f}s | tok/win={tele.get('tok_per_win')} acc={tele.get('accept_pct')}% "
              f"verwait={tele.get('ver_wait')} jobs={tele.get('jobs')} ring/tok={tele.get('ring_per_tok', float('nan'))} "
              f"| {first!r}", flush=True)
    return rows


TELE = {
    "experts": re.compile(r"experts: (\d+) requests \(([\d.]+)/token\) = RAM (\d+) hot \+ (\d+) ring, MAP (\d+), DISK (\d+) blobs / ([\d.]+) GB \(([\d.]+) GB/s\)"),
    "ring": re.compile(r"ring: (\d+) begin_layer calls, (\d+) ids seen, (\d+) hot-skipped, (\d+) fresh fetches"),
    "mtp": re.compile(r"mtp: (\d+) windows, ([\d.]+) tokens/window \(([\d.]+)% of (\d+) drafts accepted\), ([\d.]+) ms/window"),
    "phases": re.compile(r"phases/win ms: ver\[wait ([\d.]+) pool ([\d.]+) host ([\d.]+) commit ([\d.]+)\] dispatch\[plan\+disk-issue ([\d.]+) actq ([\d.]+) jobs ([\d.]+) run ([\d.]+) ringwait ([\d.]+)\] mtp ([\d.]+)"),
    "pool": re.compile(r"pool/win ms: wait_park ([\d.]+) drain ([\d.]+) repark ([\d.]+) \| native gu ([\d.]+) q ([\d.]+) down ([\d.]+)"),
    "gen": re.compile(r"prompt (\d+) tokens = .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)"),
    "cache": re.compile(r"R4 expert-cache hits"),
}


def engine_telemetry(log_path: str, tail_bytes: int = 600_000,
                      want_prompt: int = None, want_gen: int = None) -> dict:
    """Parse the per-request telemetry blocks out of the engine log's tail.

    A block = the telemetry lines the engine prints after one request.  With
    want_prompt/want_gen set, the block whose 'prompt N tokens ... G generated' line
    matches is returned (the sample's own request); otherwise the last block.
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - tail_bytes))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        return {}
    blocks = txt.split("hot-tier lookup coverage")
    blocks = [b for b in blocks if "phases/win" in b or "experts:" in b]
    if not blocks:
        return {}
    pick = blocks[-1]
    if want_prompt is not None:
        for b in reversed(blocks):
            m = list(TELE["gen"].finditer(b))
            if m:
                gpt, ggen = int(m[-1].group(1)), int(m[-1].group(2))
                if gpt == want_prompt and ggen == want_gen:
                    pick = b
                    break
    last = pick
    out = {}
    m = TELE["mtp"].search(last)
    if m:
        out["mtp_windows"] = int(m.group(1))
        out["tok_per_win"] = float(m.group(2))
        out["accept_pct"] = float(m.group(3))
        out["ms_per_win"] = float(m.group(5))
    m = TELE["experts"].search(last)
    if m:
        out["req_per_tok"] = float(m.group(2))
        out["hot"] = int(m.group(3)); out["ring"] = int(m.group(4))
        out["disk_blobs"] = int(m.group(6)); out["disk_gb"] = float(m.group(7))
        out["disk_gbs"] = float(m.group(8))
        gen_tok = out.get("mtp_windows") and None
    m = TELE["phases"].search(last)
    if m:
        out["ver_wait"], out["ver_pool"], out["ver_host"], out["commit"] = map(float, m.groups()[:4])
        out["plan"], out["actq"], out["jobs"], out["run"], out["ringwait"] = map(float, m.groups()[4:9])
    m = TELE["pool"].search(last)
    if m:
        out["drain"] = float(m.group(2)); out["gu"] = float(m.group(4)); out["down"] = float(m.group(6))
    m = list(TELE["gen"].finditer(last))
    if m:
        out["gen_ms"] = int(m[-1].group(3)); out["gen_tok"] = int(m[-1].group(2))
        out["gen_tps"] = float(m[-1].group(4))
    if out.get("gen_tok"):
        out["ring_per_tok"] = out.get("ring", 0) / out["gen_tok"]
        out["miss_blobs_per_tok"] = out.get("disk_blobs", 0) / out["gen_tok"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--port", type=int, default=8126)
    ap.add_argument("--host", default=None)
    ap.add_argument("--ctxs", default="4096")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-restart", action="store_true")
    a = ap.parse_args()

    cfgp = a.cfg if os.path.isabs(a.cfg) else os.path.join(REPO, a.cfg)
    cfg = json.load(open(cfgp))
    model = a.model or cfg.get("model_name", "m")
    log_path = cfg.get("log", "")
    if log_path and not os.path.isabs(log_path):
        log_path = os.path.join(cfg.get("cwd", REPO), log_path)
    host = a.host or os.environ.get("STRATA_HOST", "100.80.130.126")

    if not a.no_restart:
        # srv.sh's "restart" mode mis-binds PORT (it reads $2 before the shift), so
        # arm restarts go through the two documented forms: stop, then <cfg> <port>.
        print(f"[hc] stopping, then starting {a.cfg} on :{a.port}", flush=True)
        subprocess.run(["bash", os.path.join(REPO, "srv.sh"), "stop"], capture_output=True, text=True, timeout=120)
        r = subprocess.run(["bash", os.path.join(REPO, "srv.sh"), cfgp, str(a.port)],
                           capture_output=True, text=True, timeout=900)
        print(r.stdout.strip(), r.stderr.strip(), flush=True)
        if "UP after" not in r.stdout:
            print("[hc] SERVER FAILED TO COME UP")
            return 1

    arm = {"tag": a.tag, "cfg": a.cfg, "ctxs": {}}
    for ctx in [int(x) for x in a.ctxs.split(",") if x.strip()]:
        print(f"[hc] {a.tag}: ctx {ctx} x {a.n}", flush=True)
        rows = bench_ctx(host, a.port, model, ctx, a.max_tokens, a.n, log_path=log_path)
        tele = rows[-1].get("tele") if rows and rows[-1].get("tele") else (engine_telemetry(log_path) if log_path else {})
        dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"] == r["decode_tok_s"]]
        pre = [r["prefill_tok_s"] for r in rows if r["prefill_tok_s"] == r["prefill_tok_s"]]
        arm["ctxs"][ctx] = {
            "rows": [{k: v for k, v in r.items() if k != "tele"} for r in rows],
            "decode_med": statistics.median(dec) if dec else None,
            "decode_min": min(dec) if dec else None,
            "decode_max": max(dec) if dec else None,
            "prefill_med": statistics.median(pre) if pre else None,
            "telemetry": tele,
        }
        print(f"[hc]   -> decode med {arm['ctxs'][ctx]['decode_med']} "
              f"(min {arm['ctxs'][ctx]['decode_min']} max {arm['ctxs'][ctx]['decode_max']})", flush=True)

    outj = os.path.join(REPO, "bench", "results", f"hc2-{a.tag}.json")
    with open(outj, "w") as f:
        json.dump(arm, f, indent=1)
    lines = [f"### {a.tag}  ({a.cfg}, n={a.n})", "",
             "| ctx | decode med (min/max) | prefill med | tok/win | acc % | ms/win | ver wait/pool | jobs | run | ringwait | ring/tok | miss blobs/tok |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ctx, d in arm["ctxs"].items():
        t = d.get("telemetry") or {}
        lines.append(
            f"| {ctx} | {d['decode_med']:.2f} ({d['decode_min']:.2f}/{d['decode_max']:.2f}) | "
            f"{d['prefill_med'] or 0:.0f} | {t.get('tok_per_win', float('nan')):.2f} | "
            f"{t.get('accept_pct', float('nan')):.0f} | {t.get('ms_per_win', float('nan')):.1f} | "
            f"{t.get('ver_wait', float('nan')):.1f}/{t.get('ver_pool', float('nan')):.1f} | "
            f"{t.get('jobs', float('nan')):.1f} | {t.get('run', float('nan')):.1f} | "
            f"{t.get('ringwait', float('nan')):.1f} | {t.get('ring_per_tok', float('nan')):.1f} | "
            f"{t.get('miss_blobs_per_tok', float('nan')):.1f} |")
    outm = os.path.join(REPO, "bench", "results", f"hc2-{a.tag}.md")
    with open(outm, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
