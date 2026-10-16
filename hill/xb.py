#!/usr/bin/env python3
"""xb.py - diverse-prompt decode bench (held out from the profile-training corpus in hill/train_prompts.py).

  hill/xb.py --tag base --cfg hill/cfg-x.json            # restart on cfg, then bench
  hill/xb.py --tag base --cfg hill/cfg-x.json --no-restart

Decode tok/s is the engine's own number ("N generated in M ms") summed over all prompts, so
TTFT/prefill never leaks in.  Greedy sampling (server default), reasoning off, 256 tokens each.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hill"))
from hc import KEY, wait_ready, engine_telemetry  # noqa: E402

EVAL = [
    ("code-py", "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and a decorator "
                "interface. Include type hints and a short docstring per function."),
    ("code-rs", "Explain, with a Rust code example, how to build a lock-free single-producer single-consumer ring "
                "buffer using atomics. Discuss memory ordering choices."),
    ("prose", "Write a short story (about 400 words) about a lighthouse keeper who discovers that the light has "
              "been signalling to something under the sea."),
    ("explain", "Explain how the TCP congestion control algorithms Reno, CUBIC and BBR differ, and when each "
                "performs best. Be concrete."),
    ("math", "Solve step by step: a tank fills through pipe A in 6 hours and pipe B in 9 hours, and drains through "
             "pipe C in 12 hours. All three are opened at 8am with the tank empty. When is it full? Then generalise."),
    ("agent", "You are a coding agent. The user's repository has a failing test `test_parse_dates` in "
              "tests/test_util.py because util.parse_date returns naive datetimes. Plan the fix, show the exact "
              "diff to util.py and the test, and describe how you would verify it."),
]

EVAL2 = EVAL + [
    ("c-js", "Write a JavaScript debounce and throttle implementation with leading/trailing options and explain the difference with timelines."),
    ("c-sql", "Given tables orders(id, customer_id, total, created_at) and customers(id, name, country), write queries for monthly revenue by country, the top 3 customers per country, and customers with no orders in 90 days."),
    ("c-cpp", "Explain move semantics, perfect forwarding and RVO in C++ with small code examples and common pitfalls."),
    ("c-bash", "Write a robust bash backup script using rsync with exclusion lists, logging, retention of 14 daily snapshots via hard links, and error notification."),
    ("c-debug", "A Python web app intermittently returns 502 errors behind nginx under load. Walk through a systematic diagnosis covering gunicorn workers, timeouts, keepalive and resource limits."),
    ("c-arch", "Design an event-driven order-processing system with Kafka: topics, partitions, idempotency, exactly-once semantics and failure handling."),
    ("c-test", "Write Jest tests for a shopping cart module covering adding items, quantity updates, discounts, tax and edge cases."),
    ("hist", "Explain the fall of the Western Roman Empire: economic, military and political causes, and how historians' views have changed."),
    ("sci", "Explain how CRISPR-Cas9 gene editing works and discuss its main medical applications and ethical issues."),
    ("biz", "Write a go-to-market plan for a B2B SaaS tool that automates invoice reconciliation for small accountancy firms."),
    ("advice", "I want to switch careers from teaching to software development at 35. Give a realistic 12-month plan with resources and milestones."),
    ("phys", "Derive the period of a simple pendulum for small angles and discuss what changes for large amplitudes."),
    ("story2", "Write the opening of a mystery novel set in a remote Antarctic research station during the polar night."),
    ("compare", "Compare PostgreSQL, MySQL and SQLite for a new web application: features, performance, operations and when to choose each."),
]

TEMP, SEED = 1.0, 1234

GEN = re.compile(r"prompt (\d+) tokens = .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")


def chat(host, port, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": False, "reasoning_effort": "none", "temperature": TEMP, "seed": SEED}
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return time.time() - t0, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--port", type=int, default=8126)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--big", action="store_true", help="the 20-prompt held-out set")
    ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args()
    global TEMP
    TEMP = a.temp
    cfgp = a.cfg if os.path.isabs(a.cfg) else os.path.join(REPO, a.cfg)
    cfg = json.load(open(cfgp))
    model = cfg.get("model_name", "m")
    log = cfg.get("log", "")
    if log and not os.path.isabs(log):
        log = os.path.join(cfg.get("cwd", REPO), log)
    host = os.environ.get("STRATA_HOST", "127.0.0.1")
    if not a.no_restart:
        subprocess.run(["bash", os.path.join(REPO, "srv.sh"), "stop"], capture_output=True, timeout=120)
        r = subprocess.run(["bash", os.path.join(REPO, "srv.sh"), cfgp, str(a.port)], capture_output=True,
                           text=True, timeout=900)
        print(r.stdout.strip(), flush=True)
        if "UP after" not in r.stdout:
            print("[xb] SERVER FAILED", r.stderr[-2000:])
            return 1
    toks = ms = 0
    rows = []
    for rep in range(a.reps):
        for name, p in (EVAL2 if a.big else EVAL):
            if a.only and name not in a.only.split(","):
                continue
            wall, d = chat(host, a.port, model, p, a.max_tokens)
            time.sleep(0.5)
            t = engine_telemetry(log) if log else {}
            g = t.get("gen_tok"), t.get("gen_ms")
            if g[0]:
                toks += g[0]; ms += g[1]
            row = dict(name=name, rep=rep, gen=g[0], ms=g[1], tps=t.get("gen_tps"), tpw=t.get("tok_per_win"),
                       acc=t.get("accept_pct"), disk_per_tok=round(t.get("miss_blobs_per_tok", 0), 1),
                       disk_gbs=t.get("disk_gbs"), ringwait=t.get("ringwait"), verwait=t.get("ver_wait"),
                       pool=t.get("ver_pool"), mspw=t.get("ms_per_win"))
            try:
                m = d["choices"][0]["message"]
                row["text"] = (m.get("reasoning_content") or "") + (m.get("content") or "")
            except Exception:
                pass
            rows.append(row)
            print(f"  {name:8s} {row['tps']!s:>6} tok/s  tok/win {row['tpw']}  disk/tok {row['disk_per_tok']} "
                  f"@{row['disk_gbs']}GB/s  ms/win {row['mspw']} ringwait {row['ringwait']} pool {row['pool']}",
                  flush=True)
    agg = toks / (ms / 1000) if ms else 0
    nbad = sum(1 for r in rows if '!!!!!!' in r.get('text', ''))
    print(f"[xb] {a.tag}: AGG decode {agg:.2f} tok/s over {toks} tokens  DEGENERATE {nbad}/{len(rows)}", flush=True)
    os.makedirs(os.path.join(REPO, "bench", "results", "xb"), exist_ok=True)
    with open(os.path.join(REPO, "bench", "results", "xb", f"{a.tag}.json"), "w") as f:
        json.dump({"tag": a.tag, "cfg": a.cfg, "agg": agg, "rows": rows}, f, indent=1)
    with open(os.path.join(REPO, "bench", "results", "xb", "LEDGER.md"), "a") as f:
        f.write(f"| {time.strftime('%m-%d %H:%M')} | {a.tag} | {agg:.2f} | {toks} | {a.cfg} | bad {nbad} |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
