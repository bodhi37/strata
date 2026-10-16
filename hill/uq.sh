#!/bin/bash
# hill/uq.sh BASE.json "name|python-mutation" ... : for each arm build hill/cfg-NAME.json from BASE and run xb --big on the unified server
cd /home/user/strata
base=$1; shift
for spec in "$@"; do
  name=${spec%%|*}; mut=${spec#*|}
  python3 - "$base" "$name" "$mut" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); a=c['args']; e=c['env']
exec(sys.argv[3])
c['log']=f'/home/user/strata/logs/{sys.argv[2]}.log'
json.dump(c,open(f'hill/cfg-{sys.argv[2]}.json','w'),indent=1)
PY
  echo "=== $name: $mut  $(date +%T)"
  STRATA_SRV=/home/user/strata/srv-uni.sh timeout 2400 venv/bin/python hill/xb.py --big --tag $name --cfg hill/cfg-$name.json 2>&1 | tail -1
done
echo QUEUE DONE
