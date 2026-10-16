#!/bin/bash
# run xb arms in order: hill/queue.sh tag1 tag2 ...   (cfg = hill/cfg-<tag>.json), output -> logs/queue.out
cd "$(dirname "$0")/.."
for t in "$@"; do ./venv/bin/python hill/xb.py --tag "$t" --cfg "hill/cfg-$t.json" --big; done
