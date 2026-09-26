import json, time, urllib.request, sys
PORT = int(sys.argv[1]) if len(sys.argv)>1 else 8111
GEN = int(sys.argv[2]) if len(sys.argv)>2 else 200
def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(payload).encode(), {"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
# warm coherence probe
r = post("/v1/chat/completions", {"model":"x","messages":[{"role":"user","content":"Say hello and tell me what 2+2 is, briefly."}],"max_tokens":60,"temperature":0})
print("COHERENCE:", r["choices"][0]["message"]["content"][:300].replace("\n"," "))
# decode speed: short prompt, 200 new tokens
t0=time.time()
r = post("/v1/chat/completions", {"model":"x","messages":[{"role":"user","content":"Write a vivid 250-word story about a lighthouse keeper."}],"max_tokens":GEN,"temperature":0})
dt=time.time()-t0
txt = r["choices"][0]["message"]["content"]
print(f"DECODE: {GEN} tok in {dt:.1f}s = {GEN/dt:.1f} tok/s")
print("SAMPLE:", txt[:400].replace("\n"," "))
