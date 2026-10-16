import json, time, urllib.request, sys
port = int(sys.argv[1]) if len(sys.argv)>1 else 8111
ntok = int(sys.argv[2]) if len(sys.argv)>2 else 500
req = {"model":"x","messages":[{"role":"user","content":"Write a detailed essay on the history of computing, from Babbage to GPUs. Go step by step."}],"max_tokens":ntok,"temperature":0}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(), {"Content-Type":"application/json"})
t0=time.time(); resp=json.loads(urllib.request.urlopen(r, timeout=3600).read()); dt=time.time()-t0
u=resp["usage"]; print("tok",u,"time",round(dt,1),"rate",round(u["completion_tokens"]/dt,2))
