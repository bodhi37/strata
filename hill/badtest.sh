#!/bin/bash
# hill/badtest.sh cfg.json [n] - restart on cfg, run the degeneration repro prompts (greedy + temp 1), report '!!!!' hits
cd "$(dirname "$0")/.."
bash srv.sh stop >/dev/null 2>&1
bash srv.sh "$1" 8126 2>&1 | tail -1
P1="Design an event-driven order-processing system with Kafka: topics, partitions, idempotency, exactly-once semantics and failure handling."
P2="Write a short story (about 400 words) about a lighthouse keeper who discovers that the light has been signalling to something under the sea."
P3="Write a JavaScript debounce and throttle implementation with leading/trailing options and explain the difference with timelines."
bad=0; n=0
for p in "$P1" "$P2" "$P3"; do for t in 0 1.0; do
  o=$(hill/ask.sh "$p" 300 $t); n=$((n+1))
  i=$(python3 -c "import sys;t=sys.stdin.read();print(t.find('!!!!'))" <<< "$o")
  [ "$i" != "-1" ] && bad=$((bad+1))
  echo "  temp $t len ${#o} bad_at $i"
done; done
echo "RESULT $1: $bad/$n degenerate"
