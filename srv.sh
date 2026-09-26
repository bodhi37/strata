#!/bin/bash
# R10: durable server control.  The old habit - pkill -f "engine/strata --serve" from a bash -c block -
# killed the block's own shell (the pattern matches the wrapper's command line), which is how several
# launches "vanished".  These scripts keep the pattern out of the caller's command line via [.] classes
# and run the server under tmux, which survives tool-call blocks.
set -u
cd "$(dirname "$0")"
PORT="${2:-8111}"
CFG="$1"
SESSION="srv${PORT}"

stop_server() {
  tmux kill-session -t "$SESSION" 2>/dev/null
  for pid in $(pgrep -f 'serve[r]/server\.py'); do kill "$pid" 2>/dev/null; done
  for pid in $(pgrep -f 'strat[a]/engine/strata '); do kill "$pid" 2>/dev/null; done
  sleep 3
  for pid in $(pgrep -f 'strat[a]/engine/strata '); do kill -9 "$pid" 2>/dev/null; done
}

if [[ "$1" == "stop" ]]; then stop_server; echo stopped; exit 0; fi
if [[ "$1" == "restart" ]]; then stop_server; shift; CFG="$1"; fi

[[ -f "$CFG" ]] || { echo "no config: $CFG"; exit 1; }
stop_server
mkdir -p logs
tmux new-session -d -s "$SESSION" \
  "./venv/bin/python serve/server.py --engine strata --config '$CFG' --port '$PORT' 2>&1 | tee -a logs/srv-$(basename '$CFG' .json).log.srv"
# wait for readiness
for i in $(seq 1 90); do
  sleep 5
  if curl -s --max-time 3 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q qwen; then
    echo "UP after $((i*5))s on :$PORT with $CFG"
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then echo "SERVER DIED during startup"; exit 1; fi
done
echo "TIMEOUT waiting for readiness"
exit 1
