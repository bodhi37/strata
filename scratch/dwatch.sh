#!/bin/bash
# $1 = seconds to watch; prints device read stats delta
DEV=nvme0n1
D=${1:-10}
R0=($(grep " $DEV " /proc/diskstats | awk '{print $4, $6, $7}'))
T0=$(date +%s.%N)
sleep "$D"
R1=($(grep " $DEV " /proc/diskstats | awk '{print $4, $6, $7}'))
T1=$(date +%s.%N)
DT=$(echo "$T1 - $T0" | bc)
OPS=$((R1[0]-R0[0])); SEC=$(( (R1[1]-R0[1])/2 )); MS=$((R1[2]-R0[2]))
echo "window ${DT}s: read_ops=$OPS MB=$((SEC/1000)) eff=$((SEC/1000*1000/(SEC/1000+1))) avg_MB_per_op=$((SEC/(OPS+1)/2))K"
echo "device_read_busy=${MS}ms of $((D*1000))ms = $((MS*100/(D*1000)))% busy; MB/s=$(echo "scale=1; $SEC/1000/$DT" | bc)"
