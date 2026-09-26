import json, time, urllib.request, sys
PORT = int(sys.argv[1]) if len(sys.argv)>1 else 8111
NTOK = int(sys.argv[2]) if len(sys.argv)>2 else 4096
# a prompt of ~NTOK tokens: repeated technical paragraph
para = ("The lighthouse keeper surveyed the horizon: lens mechanics, tide tables, and the logbook of storms. "
        "He adjusted the brass gears, checked the mercury bath, and recorded the barometric pressure at dusk. ")
prompt = para * (NTOK // 45 + 1)
def post(path, payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(payload).encode(), {"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
t0=time.time()
r = post("/v1/chat/completions", {"model":"x","messages":[{"role":"user","content":prompt+"\n\nSummarize the keeper's routine in one sentence."}],"max_tokens":80,"temperature":0})
dt=time.time()-t0
u = r.get("usage", {})
print(f"PROMPT-TOK={u.get('prompt_tokens')}  total={dt:.1f}s  prefill~={u.get('prompt_tokens',0)/dt:.1f} tok/s  ans={((r['choices'][0]['message'].get('content') or '')[:120])!r}")
