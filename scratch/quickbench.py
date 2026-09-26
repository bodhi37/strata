import json, time, urllib.request, sys
PORT = int(sys.argv[1]) if len(sys.argv)>1 else 8111
GEN = int(sys.argv[2]) if len(sys.argv)>2 else 300
def post(path, payload, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(payload).encode(), {"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
def show(r, n=260):
    m = r["choices"][0]["message"]
    txt = m.get("content") or ""
    think = m.get("reasoning_content") or ""
    return (txt or think)[:n].replace("\n"," "), len(txt), len(think)
r = post("/v1/chat/completions", {"model":"x","messages":[{"role":"user","content":"Say hello and tell me what 2+2 is, briefly. Answer directly."}],"max_tokens":500,"temperature":0})
s, a, b = show(r); print("COHERENCE:", s, f"[answer {a} ch, think {b} ch]")
t0=time.time()
r = post("/v1/chat/completions", {"model":"x","messages":[{"role":"user","content":"Write a vivid 200-word story about a lighthouse keeper. No preamble."}],"max_tokens":GEN,"temperature":0})
dt=time.time()-t0
s, a, b = show(r)
usage = r.get("usage", {})
comp = usage.get("completion_tokens", GEN)
print(f"DECODE: {comp} tok in {dt:.1f}s = {comp/dt:.1f} tok/s")
print("SAMPLE:", s[:400])
