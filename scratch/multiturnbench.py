import json, time, urllib.request, sys
PORT = int(sys.argv[1]) if len(sys.argv)>1 else 8111
def post(payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
para = ("The lighthouse keeper surveyed the horizon: lens mechanics, tide tables, and the logbook of storms. "
        "He adjusted the brass gears, checked the mercury bath, and recorded the barometric pressure at dusk. ") * 90
history = []
for turn, q in enumerate(["Summarize the keeper's routine in one sentence.",
                          "What instrument did he check at dusk? Answer in one short sentence.",
                          "Name the three things he maintains. One short sentence."]):
    history.append({"role":"user","content": (para if turn==0 else "") + q})
    t0=time.time()
    r = post({"model":"x","messages":list(history),"max_tokens":120,"temperature":0})
    dt=time.time()-t0
    u=r.get("usage",{})
    txt=(r["choices"][0]["message"].get("content") or r["choices"][0]["message"].get("reasoning_content") or "")[:90]
    print(f"turn {turn}: prompt={u.get('prompt_tokens')} total={dt:.1f}s prefill~={u.get('prompt_tokens',0)/dt:.0f} tok/s  fin={r['choices'][0].get('finish_reason')}  ans={txt!r}")
    history.append({"role":"assistant","content": r["choices"][0]["message"].get("content") or ""})
