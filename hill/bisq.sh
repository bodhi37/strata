#!/bin/bash
cd "$(dirname "$0")/.."
for t in "$@"; do hill/badtest.sh hill/cfg-bis-$t.json; done
