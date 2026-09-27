import json, time, urllib.request, sys
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8111

def post(payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def ask(name, content, max_tokens=150):
    t0 = time.time()
    r = post({"model": "x", "messages": [{"role": "user", "content": content}],
              "max_tokens": max_tokens, "temperature": 0})
    dt = time.time() - t0
    u = r.get("usage", {})
    m = r["choices"][0]["message"]
    txt = (m.get("content") or m.get("reasoning_content") or "")
    ptok = u.get("prompt_tokens", 0)
    ctok = u.get("completion_tokens", 0)
    gen_tps = ctok / max(dt - ptok / max(u.get("prefill_tokens_per_s", 0), 1), 1e-9) if False else None
    print(f"--- {name}: prompt={ptok} tok, {dt:.1f}s total, finish={r['choices'][0].get('finish_reason')}")
    print(repr(txt[:300]))
    return txt

# 1. short
ask("SHORT", "What is the capital of France? Answer in one sentence.", 60)
# 2. medium
ask("MEDIUM", "Write a 100-word story about a robot learning to paint.", 200)
# 3. long prefill ~ 12k tokens then decode
para = ("The lighthouse keeper surveyed the horizon: lens mechanics, tide tables, and the logbook of storms. "
        "He adjusted the brass gears, checked the mercury bath, and recorded the barometric pressure at dusk. ") * 155
ask("LONG-PREFILL", para + "\n\nIgnore the passage above. What is the capital of France? Answer in one sentence.", 60)
# 4. long prefill + real task (reading comprehension) to see if state is corrupted
ask("LONG-TASK", para + "\n\nWhat three things does the keeper maintain? Answer briefly.", 100)
