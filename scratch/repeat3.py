import json, time, urllib.request
req = {"model":"x","messages":[{"role":"user","content":"Explain how a bill becomes law in the United States, step by step."}],"max_tokens":300,"temperature":0}
for i in range(3):
    r = urllib.request.Request("http://127.0.0.1:8111/v1/chat/completions", json.dumps(req).encode(), {"Content-Type":"application/json"})
    t0=time.time(); resp=json.loads(urllib.request.urlopen(r, timeout=3600).read()); dt=time.time()-t0
    u=resp["usage"]; print(f"run{i}: gen={u['completion_tokens']} t={dt:.1f} rate={u['completion_tokens']/dt:.2f}")
