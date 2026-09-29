"""R15 ground-truth probe: what the DRIVE actually does during a request.

/engine telemetry mixes prefill staging reads and decode ring reads into one `DISK` counter, so the only
honest number for "is decode drive-bound" is /proc/diskstats around the request.  This wraps a request with
diskstats + a tier-state read and prints MB, MB/s, io-busy%, per-token bytes and the engine's own lines.

    python3 scratch/r15_probe.py 8123 [ntok] [prompt-key|all] [tag]
"""
import json, time, urllib.request, sys, re, os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WHICH = sys.argv[3] if len(sys.argv) > 3 else "all"
TAG = sys.argv[4] if len(sys.argv) > 4 else ""

DEV = "nvme0n1"
LOG = os.environ.get("R15_LOG", "")

TOPICS = {
    "cooking":  "Explain how sourdough fermentation works, covering the roles of yeast and lactic acid bacteria.",
    "coding":   "Write a Python function that merges two sorted lists into one sorted list without duplicates, and explain it.",
    "law":      "Explain how a bill becomes law in the United States, step by step.",
    "medicine": "Explain how mRNA vaccines work.",
    "math":     "Prove that the square root of 2 is irrational.",
    "agentic":  "You are debugging a web service returning 500 errors under load. Walk through a systematic diagnosis plan.",
}


def disk():
    for ln in open("/proc/diskstats"):
        f = ln.split()
        if f[2] == DEV:
            return dict(rd_ops=int(f[3]), rd_kb=int(f[5]) // 2, wr_ops=int(f[6]), wr_kb=int(f[8]) // 2,
                        ms_io=int(f[9]), inflight=int(f[11]))
    raise SystemExit("no " + DEV)


def post(payload, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def logmark():
    if not LOG or not os.path.exists(LOG):
        return 0
    return os.path.getsize(LOG)


def logtail(mark):
    if not LOG or not os.path.exists(LOG):
        return ""
    with open(LOG, "rb") as f:
        f.seek(mark)
        return f.read().decode("utf8", "replace")


def run(name, q):
    a, t0 = disk(), time.time()
    r = post({"model": "x", "messages": [{"role": "user", "content": q}],
              "max_tokens": NTOK, "temperature": 0})
    dt = time.time() - t0
    b = disk()
    u = r.get("usage", {})
    comp = u.get("completion_tokens", 0) or 0
    ptok = u.get("prompt_tokens", 0) or u.get("input_tokens", 0) or 0
    m = r["choices"][0]["message"]
    txt = (m.get("content") or "") + "\x01" + (m.get("reasoning_content") or "")
    ok = "!!!" not in txt[:400] and len(txt) > 50
    rkb, wkb = b["rd_kb"] - a["rd_kb"], b["wr_kb"] - a["wr_kb"]
    busy = 100.0 * (b["ms_io"] - a["ms_io"]) / (dt * 1000)
    print(f"{TAG}{name:9s} prompt {ptok:6d} comp {comp:5d} {dt:7.1f}s decode {comp/dt:5.2f} tok/s "
          f"| DISK rd {rkb/1e6:7.2f} GB ({rkb/1e3/dt:6.1f} MB/s) wr {wkb/1e6:6.2f} GB busy {busy:5.1f}% "
          f"| {rkb*1024/max(comp,1)/1e6:6.2f} MB/tok {'OK' if ok else 'GARBAGE'}")
    return dict(name=name, comp=comp, dt=dt, rkb=rkb, busy=busy, ptok=ptok, ok=ok)


names = list(TOPICS) if WHICH == "all" else [WHICH]
res = []
for n in names:
    mark = logmark()
    res.append(run(n, TOPICS[n]))
    tl = logtail(mark)
    for pat in (r"experts: .*", r"ring: .*", r"mtp: .*", r"phases/win ms: .*", r"prompt .*", r"lru: .*"):
        for mm in re.findall(pat, tl)[-1:]:
            print("   ! " + mm.strip())
rates = sorted(x["dt"] / max(x["comp"], 1) for x in res if x["comp"])
if rates:
    r0 = sum(rates) / len(rates)
    print(f"{TAG}AVG {1/r0:5.2f} tok/s  |  drive {sum(x['rkb'] for x in res)/1e3/sum(x['dt'] for x in res):6.1f} MB/s"
          f"  busy {sum(x['busy'] for x in res)/len(res):5.1f}%")
