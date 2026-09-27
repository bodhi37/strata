import json, time, urllib.request, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
TARGET_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 400
LABEL = sys.argv[4] if len(sys.argv) > 4 else "ctx"

# Build a synthetic long-context document: a structured "deployment runbook" of
# many sections with unique keys, with needles planted at 10% / 50% / 90% depth.
# The task requires integrating three needles + a reasoning step, so a model that
# lost the middle of the document cannot answer correctly.
NEEDLES = [
    ("MARGIN-CODE-7743", "the rollback margin for the billing replica set is exactly 14.2 percent"),
    ("OPS-QUOTA-2291", "the drain quota for the ingest fleet is exactly 38 concurrent jobs"),
    ("AUDIT-KEY-5580", "the retention deadline for the audit ledger is exactly 2027-03-19"),
]

def build_prompt(tok):
    # rough token estimate: ~1.3 tokens per word for this style
    words_needed = int(TARGET_TOKENS / 1.3)
    chunk = ("Section {i}: The deployment runbook for cluster {i} covers node provisioning, "
             "storage quotas, replica placement, failover drills, and alert thresholds. "
             "Operators must reconcile the desired state with the observed state every "
             "maintenance window, record drift in the change log, and escalate any "
             "divergence that persists beyond two consecutive windows.\n")
    out = []
    n = 0
    i = 0
    total_words = words_needed
    while n < total_words:
        i += 1
        out.append(chunk.format(i=i))
        n += len(chunk.format(i=i).split())
        for key, fact in NEEDLES:
            if abs(n - total_words * (0.1 + 0.4 * NEEDLES.index((key, fact)))) < 60:
                out.append(f"Special provision [{key}]: note that {fact}.\n")
                n += 18
    body = "".join(out)
    q = (body +
         "\n\nAnswer with three short lines only:\n"
         "1. The rollback margin for the billing replica set.\n"
         "2. The drain quota for the ingest fleet.\n"
         "3. The retention deadline for the audit ledger.\n"
         "Then one final line starting with CHECK: explaining how these three constraints "
         "interact during a failover drill (one sentence).")
    return q

def post(payload, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

q = build_prompt(TARGET_TOKENS)
t0 = time.time()
r = post({"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": NTOK, "temperature": 0})
dt = time.time() - t0
u = r.get("usage", {})
pt, comp = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
m = r["choices"][0]["message"]
txt = (m.get("content") or "") + "\x01" + (m.get("reasoning_content") or "")
ok = all(s in txt for s in ("14.2", "38", "2027-03-19")) and "!!!" not in txt
print(f"{LABEL}: prompt={pt} tok, completion={comp} tok, total {dt:.1f}s")
print(f"  aggregate {comp/dt:.2f} tok/s (includes prefill); prefill-only ~{pt/(dt - comp*0.13):.0f} tok/s (decode ~7.7 tok/s est)")
print(f"  needles {'ALL FOUND' if ok else 'NOT VISIBLE IN BUDGET'}: {txt[:300].replace(chr(10), ' | ')}")
