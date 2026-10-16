#!/bin/bash
# sample diskstats while a reader runs
DEV=nvme0n1
read a b <<< "$(grep " $DEV " /proc/diskstats | awk '{print $4, $8}')"
for i in $(seq 1 60); do sleep 0.25; done &
wait $!
read c d <<< "$(grep " $DEV " /proc/diskstats | awk '{print $4, $8}')"
echo "ops=$((c-a)) busy_ms=$((d-b))"
