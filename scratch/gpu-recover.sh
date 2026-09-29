#!/bin/bash
# GPU recovery for the 2026-09-29 11:12 OOM wedge.
# The OOM killer killed the running engine (anon-rss 25.2 GB); the killed CUDA context
# left a stuck channel in the kernel driver (NV_ERR_STATE_IN_USE at the GSP), so no new
# CUDA context can be created while the display path still works.  This script bounces
# the desktop, reloads the NVIDIA stack, re-enables persistence mode and verifies CUDA
# with scratch/cuda_probe.  Run as root:  sudo bash scratch/gpu-recover.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/gpu-recover.log"
mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }
holders() { fuser /dev/nvidia0 /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u; }

log "==== gpu-recover start ===="
log "sessions: $(loginctl list-sessions --no-legend 2>/dev/null | tr '\n' ';')"

log "stopping display-manager"
systemctl stop display-manager
sleep 5

H="$(holders)"
if [ -n "$H" ]; then
  log "GPU holders after sddm stop: $(for p in $H; do echo -n "$p($(ps -o comm= -p $p)) "; done)"
  SID="$(loginctl list-sessions --no-legend 2>/dev/null | awk '$4=="seat0"{print $1; exit}')"
  if [ -n "$SID" ]; then log "terminating logind session $SID"; loginctl terminate-session "$SID"; sleep 5; fi
fi
H="$(holders)"
if [ -n "$H" ]; then
  log "still held; killing: $H"
  kill $H 2>/dev/null
  sleep 3
  kill -9 $H 2>/dev/null
  sleep 2
fi

for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
  ok=0
  for t in 1 2 3; do
    if ! lsmod | grep -q "^$m "; then ok=1; break; fi
    if modprobe -r "$m" 2>/dev/null; then log "unloaded $m"; ok=1; break; fi
    log "retry $t: $m busy"; sleep 3
  done
  if [ "$ok" != 1 ]; then
    log "FATAL: $m still loaded; aborting (restarting desktop)"
    systemctl start display-manager
    exit 1
  fi
done

for m in nvidia nvidia_uvm nvidia_modeset nvidia_drm; do
  if modprobe "$m" 2>/dev/null; then log "loaded $m"; else
    log "FATAL: cannot load $m"; systemctl start display-manager; exit 1
  fi
done

nvidia-smi -pm 1 >/dev/null 2>&1
log "persistence: $(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)  modeset: $(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null)"
if [ -c /dev/nvidia-uvm ]; then log "/dev/nvidia-uvm present"; else log "WARN: /dev/nvidia-uvm missing"; fi

PROBE="$ROOT/scratch/cuda_probe"
if sudo -u bodhi "$PROBE" 2>/dev/null | grep -q 'hostAlloc 644MiB mapped: no error'; then
  log "CUDA probe: OK"
else
  log "CUDA probe: FAILED"
  sudo -u bodhi "$PROBE" 2>&1 | sed 's/^/  /'
  systemctl start display-manager
  exit 1
fi

systemctl start display-manager
log "sddm restarted; gpu-recover done"
