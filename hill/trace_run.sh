#!/bin/bash
# train-corpus trace + counts, then eval-prompt trace + counts, on the dev engine
cd /home/user/strata
rm -f data/trace-train.bin data/trace-eval.bin
./venv/bin/python hill/collect_counts.py --cfg hill/cfg-trace.json --restart --max-tokens 320 > logs/collect-trace.out 2>&1
python3 - <<'PY'
import json
c=json.load(open('/home/user/strata/hill/cfg-trace.json'))
c['env']['STRATA_ROUTE_TRACE']='/home/user/strata/data/trace-eval.bin'
c['args'][c['args'].index('--dump-counts')+1]='/home/user/strata/data/counts-sc117-x2eval.strc'
c['log']='logs/hc2-trace-eval.log'
json.dump(c,open('/home/user/strata/hill/cfg-trace-eval.json','w'),indent=1)
PY
./venv/bin/python hill/xb.py --tag trace-eval --cfg hill/cfg-trace-eval.json > logs/xb-trace-eval.out 2>&1
echo TRACE_DONE
