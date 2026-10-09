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
# Bind to the Tailscale address (not 0.0.0.0) so the model is reachable from the laptop
# over the tailnet without being exposed on the LAN. Auth comes from a 0600 key file because
# this box has no passwordless sudo, so the firewall cannot be used as a second layer.
TS_IP="${STRATA_HOST:-100.80.130.126}"
API_KEY="$(cat "$HOME/.config/qwen-serve/api-key" 2>/dev/null || true)"

stop_server() {
  tmux kill-session -t "$SESSION" 2>/dev/null
  for pid in $(pgrep -f 'serv[e]/server\.py'); do kill "$pid" 2>/dev/null; done
  for pid in $(pgrep -f 'strat[a]/engine/strata '); do kill "$pid" 2>/dev/null; done
  sleep 3
  for pid in $(pgrep -f 'strat[a]/engine/strata '); do kill -9 "$pid" 2>/dev/null; done
}

if [[ "$1" == "stop" ]]; then stop_server; echo stopped; exit 0; fi
if [[ "$1" == "restart" ]]; then stop_server; shift; CFG="$1"; fi

[[ -f "$CFG" ]] || { echo "no config: $CFG"; exit 1; }
stop_server
mkdir -p logs
# Resolve the log path here, in bash: a `$(basename '$CFG')` inside the tmux string leaves a literal
# $CFG that the shell running the command expands to EMPTY (fish), and tee then logs nowhere.
LOGF="logs/srv-$(basename "$CFG" .json).log.srv"
# 300 s of engine silence = a stalled request, not a slow one (a 16k prefill chunk runs 60-90 s here,
# 2x-4x margin); 180 (server default) is one QLC/zram hiccup away from killing a HEALTHY engine and
# losing every cache. 0 disables.
WATCHDOG_S="${STRATA_WATCHDOG_S:-300}"
# R22: the memory governor's floor (MiB of MemAvailable).  Under it, between requests, the server tells the
# engine to shed cold-scored slices of the mlocked hot tier instead of letting the box swap-storm to where
# CUDA allocations die (the 2026-10-04/05 cublasCreate deaths); 2.5 GiB above it, shed slices regrow and
# DMA comes back.  STRATA_MEM_FLOOR_MIB=0 disables the governor entirely (a nop for A/B arms).
MEM_FLOOR_MIB="${STRATA_MEM_FLOOR_MIB:-1536}"
tmux new-session -d -s "$SESSION" -e STRATA_API_KEY="$API_KEY" -e STRATA_WATCHDOG_S="$WATCHDOG_S" \
  -e STRATA_MEM_FLOOR_MIB="$MEM_FLOOR_MIB" \
  "./venv/bin/python serve/server.py --engine strata --config '$CFG' --port '$PORT' --host '$TS_IP' 2>&1 | tee -a $LOGF"
# wait for readiness
for i in $(seq 1 90); do
  sleep 5
  if curl -s --max-time 3 -H "Authorization: Bearer $API_KEY" "http://${TS_IP}:${PORT}/v1/models" 2>/dev/null | grep -q qwen; then
    echo "UP after $((i*5))s on :$PORT with $CFG"
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then echo "SERVER DIED during startup"; exit 1; fi
done
echo "TIMEOUT waiting for readiness"
exit 1
