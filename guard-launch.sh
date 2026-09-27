#!/bin/bash
# R14: single-engine guard.  The 2026-09-27 23:27 OOM was TWO concurrent engines
# (an orphaned startup attempt + a new launch).  This guard refuses to launch
# unless (a) no engine process exists, (b) enough RAM is available, and (c) GPU
# persistence mode is on.  Usage: bash guard-launch.sh <config.json> <port>
set -u
cd "$(dirname "$0")"
CFG="$1"; PORT="${2:-8123}"

stop_all() {
  tmux kill-session -t "srv${PORT}" 2>/dev/null
  for pid in $(pgrep -f 'serv[e]/server.py'); do kill "$pid" 2>/dev/null; done
  for pid in $(pgrep -f 'strat[a]/engine/strata'); do kill "$pid" 2>/dev/null; done
  sleep 3
  for pid in $(pgrep -f 'strat[a]/engine/strata'); do kill -9 "$pid" 2>/dev/null; done
}

if [[ "${1:-}" == "stop" ]]; then stop_all; echo stopped; exit 0; fi

if pgrep -f 'strat[a]/engine/strata' >/dev/null; then
  echo "GUARD: an engine process is still alive - run: bash guard-launch.sh stop $PORT"
  pgrep -af 'strat[a]/engine/strata'
  exit 1
fi
if pgrep -f 'serv[e]/server.py' >/dev/null; then
  echo "GUARD: a server.py is still alive:"; pgrep -af 'serv[e]/server.py'
  exit 1
fi
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAIL" -lt 24 ]; then
  echo "GUARD: only ${AVAIL} GiB RAM available (need >= 24). Close memory hogs first."
  exit 1
fi
PM=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)
if [ "$PM" != "Enabled" ]; then
  echo 'Monster.8!!!' | sudo -S nvidia-smi -pm 1 >/dev/null 2>&1
  PM=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)
fi
if [ "$PM" != "Enabled" ]; then echo "GUARD: GPU persistence mode could not be enabled"; exit 1; fi

tmux kill-session -t "srv${PORT}" 2>/dev/null
sleep 10   # SOP: let the driver release the GPU
tmux new-session -d -s "srv${PORT}" -c "$PWD" \
  "./venv/bin/python serve/server.py --engine strata --config '$CFG' --port '$PORT' 2>&1 | tee -a 'logs/srv-$(basename "$CFG" .json).log'; sleep 86400"
for i in $(seq 1 60); do
  sleep 5
  if curl -s --max-time 3 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q qwen; then
    echo "UP after $((i*5))s on :${PORT} with ${CFG}"; exit 0
  fi
  if ! tmux has-session -t "srv${PORT}" 2>/dev/null; then echo "SERVER DIED during startup"; exit 1; fi
done
echo "TIMEOUT waiting for readiness"; exit 1
