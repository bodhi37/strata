import json, time, urllib.request, sys, os

# Host + auth come from the env so this works against the Tailscale-bound live endpoint.
# Defaults keep the old behaviour (localhost, no auth) when both are unset.
HOST = os.environ.get("STRATA_BENCH_HOST", "127.0.0.1")
KEY = os.environ.get("STRATA_BENCH_KEY", "")

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 200
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LABEL = sys.argv[4] if len(sys.argv) > 4 else "mt"

# Multi-turn agentic pattern: the client resends the FULL conversation every turn
# (what Pi/OpenAI/Anthropic harnesses do), so the engine's conversation cache and
# the expert tier's warmth carry across turns.  Turn 1 is a cold-ish reference;
# turns 2+ measure the resumed path.
TOPICS = [
    ("agentic",  [
        "You are debugging a web service returning 500 errors under load. Walk through a systematic diagnosis plan.",
        "Continue: now the metrics show p99 latency spiking on the database host. What next?",
        "Now write the postmortem outline for the incident, assuming the root cause was connection-pool exhaustion.",
    ]),
    ("coding",   [
        "Write a Python function that merges two sorted lists into one sorted list without duplicates, and explain it.",
        "Now extend it to k sorted lists and give the complexity.",
        "Now add type hints and doctests to the k-way version.",
    ]),
    ("math",     [
        "Prove that the square root of 2 is irrational.",
        "Adapt the argument to prove that cube root of 3 is irrational.",
        "Where exactly does the argument fail for the square root of 4?",
    ]),
]

def post(payload, timeout=3600):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions",
                                 json.dumps(payload).encode(), headers)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

all_gen23 = []
for name, turns in TOPICS[:TURNS and len(TOPICS)]:
    hist = []
    for t, q in enumerate(turns):
        hist.append({"role": "user", "content": q})
        t0 = time.time()
        r = post({"model": "x", "messages": hist, "max_tokens": NTOK, "temperature": 0})
        dt = time.time() - t0
        u = r.get("usage", {})
        comp = u.get("completion_tokens", 0)
        prompt = u.get("prompt_tokens", 0)
        txt = (r["choices"][0]["message"].get("content") or "") + "\x01" + (r["choices"][0]["message"].get("reasoning_content") or "")
        ok = "!!!" not in txt[:400] and len(txt) > 50
        all_gen23.append((comp, dt))
        tag = "t1" if t == 0 else "t2+"
        if t > 0: all_gen23[-1] = all_gen23[-1]  # keep all; summary splits below
        print(f"{LABEL} {name:9s} turn{t+1}: prompt={prompt:6d} comp={comp:4d} tok {dt:6.1f}s {comp/dt:5.2f} tok/s {'OK' if ok else 'GARBAGE'} | {txt[:80].replace(chr(10),' ').replace(chr(1),' | ')}")
        hist.append({"role": "assistant", "content": (r["choices"][0]["message"].get("content") or "")})

print(f"{LABEL} summary: all generations above; turn1 is the prefill-dominated reference, turns 2+ show the resumed decode rate")
