import json, urllib.request, sys
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8111
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 120
Q = "Explain how a bill becomes law in the United States, step by step. Be concise."
def post(payload, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
r = post({"model": "x", "messages": [{"role": "user", "content": Q}], "max_tokens": NTOK, "temperature": 0})
m = r["choices"][0]["message"]
txt = (m.get("content") or "") + "\x01" + (m.get("reasoning_content") or "")
h = hash(txt) & 0xffffffff
print(f"tokens={r.get('usage',{}).get('completion_tokens')} hash={h:08x}")
print(txt[:600].replace("\x01", "\n[THINK]\n"))
