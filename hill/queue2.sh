#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "hill/queu[e].sh" >/dev/null; do sleep 10; done
exec hill/queue.sh "$@"
