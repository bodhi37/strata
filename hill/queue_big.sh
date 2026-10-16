#!/bin/bash
cd /home/user/strata
for arm in "A:hill/cfg-x0.json" "B:hill/cfg-B.json" "C4:hill/cfg-C4.json" "C8:hill/cfg-C8.json" "C12:hill/cfg-C12.json"; do
  tag=${arm%%:*}; cfg=${arm#*:}
  ./venv/bin/python hill/xb.py --tag big-$tag --cfg $cfg --big --max-tokens 300 > logs/xb-big-$tag.out 2>&1
  grep AGG logs/xb-big-$tag.out
done
echo QUEUE_DONE
