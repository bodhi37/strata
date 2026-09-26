#!/bin/bash
# resume_pulls.sh - restart the IQ3_M/IQ4_XS pulls once the IQ3_XXS shard-4 (critical path) lands.
ROOT=/home/bodhi/models/strata
F=$ROOT/model/Qwen3.8-Flash-Next-heretic-2-IQ3_XXS-00004-of-00005.gguf
W=47528931424
until [ -f "$F" ] && [ "$(stat -c%s "$F" 2>/dev/null)" = "$W" ]; do sleep 30; done
echo "IQ3_XXS shard4 complete $(date -Is); resuming quant pulls"
setsid nohup bash $ROOT/pull_quant.sh IQ3_M  > $ROOT/logs/pull-iq3m.log  2>&1 < /dev/null &
setsid nohup bash $ROOT/pull_quant.sh IQ4_XS > $ROOT/logs/pull-iq4xs.log 2>&1 < /dev/null &
sleep 10
# also restart the parallel direct pull of the IQ3_M expert shard (the puller does 00005 first)
setsid nohup rsync -a --partial --info=progress2 \
  x64:/mnt/1tbssd/ssh-downloads/aimodels/qwen38-fn-heretic2-iq3-m/IQ3_M/Qwen3.8-Flash-Next-heretic-2-IQ3_M-00004-of-00005.gguf \
  $ROOT/model/ > $ROOT/logs/rsync-iq3m-shard4.log 2>&1 < /dev/null &
echo "resumed $(date -Is)"