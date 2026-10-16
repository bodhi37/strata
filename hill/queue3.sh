#!/bin/bash
# wait until no other queue or bench runs (two idle checks 20 s apart), then run the arms
cd "$(dirname "$0")/.."
idle() { ! pgrep -f "hill/queu[e].sh|hill/queu[e]2.sh|hill/x[b].py" >/dev/null; }
while true; do if idle; then sleep 20; idle && break; fi; sleep 10; done
exec hill/queue.sh "$@"
