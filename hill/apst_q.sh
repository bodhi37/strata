#!/bin/bash
cd "$(dirname "$0")/.."
./venv/bin/python hill/xb.py --tag aw --cfg hill/cfg-aw.json --big
sudo sh -c "printf 0 > /sys/class/nvme/nvme0/power/pm_qos_latency_tolerance_us"
cat /sys/class/nvme/nvme0/power/pm_qos_latency_tolerance_us
./venv/bin/python hill/xb.py --tag awp --cfg hill/cfg-awp.json --big
