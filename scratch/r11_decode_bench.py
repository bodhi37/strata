import json, time, urllib.request, sys
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8111
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 200
TOPICS = [
    ("cooking",  "Explain how sourdough fermentation works, covering the roles of yeast and lactic acid bacteria."),
    ("coding",   "Write a Python function that merges two sorted lists into one sorted list without duplicates, and explain it."),
    ("law",      "Explain how a bill becomes law in the United States, step by step."),
    ("medicine", "Explain how mRNA vaccines work."),
    ("math",     "Prove that the square root of 2 is irrational."),
    ("agentic",  "You are debugging a web service returning 500 errors under load. Walk through a systematic diagnosis plan."),
]
def post(payload, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

run = sys.argv[3] if len(sys.argv) > 3 else "warm"
res = []
for name, q in TOPICS:
    t0 = time.time()
    r = post({"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": NTOK, "temperature": 0})
    dt = time.time() - t0
    u = r.get("usage", {})
    comp = u.get("completion_tokens", 0)
    m = r["choices"][0]["message"]
    txt = (m.get("content") or "") + "\x01" + (m.get("reasoning_content") or "")
    ok = "!!!" not in txt[:400] and len(txt) > 50
    res.append((name, comp, dt, comp/dt, ok, txt[:110].replace("\n", " ").replace("\x01", " | ")))
    print(f"{run} {name:9s}: {comp:4d} tok {dt:6.1f}s {comp/dt:5.2f} tok/s {'OK' if ok else 'GARBAGE'} | {res[-1][5][:90]}")
rates = [x[3] for x in res]
rates.sort()
print(f"{run}: min {rates[0]:.2f}  median {rates[len(rates)//2]:.2f}  max {rates[-1]:.2f} tok/s")
