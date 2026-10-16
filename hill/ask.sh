#!/bin/bash
# hill/ask.sh "prompt" [max_tokens] [temp] - one chat against the running endpoint, prints the reply
KEY=$(cd "$(dirname "$0")/.." && ./venv/bin/python -c "import sys;sys.path.insert(0,'hill');from hc import KEY;print(KEY)")
P=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1")
curl -s http://${STRATA_HOST:-127.0.0.1}:8126/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":$P}],\"max_tokens\":${2:-120},\"temperature\":${3:-0},\"seed\":1234,\"reasoning_effort\":\"none\"}" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);m=d['choices'][0]['message'];print((m.get('reasoning_content') or '')+(m.get('content') or ''))"
