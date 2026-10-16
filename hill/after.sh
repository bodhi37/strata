#!/bin/bash
# hill/after.sh LOG OUT BASE arms... : wait for LOG to say QUEUE DONE, then run hill/uq.sh BASE arms > OUT
cd /home/user/strata
log=$1; out=$2; shift 2
until grep -q "QUEUE DONE" "$log" 2>/dev/null; do sleep 15; done
exec bash hill/uq.sh "$@" > "$out" 2>&1
