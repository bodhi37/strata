### IQ3_XXS  (coherence gate: FAIL 4 / 8; full answers in bench/results/r7-IQ3_XXS-eval.json)

config: `--expert-cache -2 --hot-ram-gib 24 --mmap-experts --pool-workers 12 --prefill 2048 --spec 4 --spec-min-p 0.8 --mtp mtp/rt --kv int8 --expert-profile data/profile-r7.bin`

| context | prompt tok | TTFT s | prefill tok/s | decode tok/s | completion tok |
| --- | ---: | ---: | ---: | ---: | ---: |
| ~1024 | 1099 | 31.8 | 35 | 6.39 | 192 |
| ~4096 | 4206 | 81.8 | 51 | 5.24 | 192 |
| ~32768 | 33260 | 498.3 | 67 | 7.48 | 192 |
