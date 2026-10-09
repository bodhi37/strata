### sc117-q2-final  (strata-sc117-q2-final.json — Q2_0, 21 GiB hot, pool 12, prefill 16384, spec-6, q4_0 KV, prompt-cache 6)

Config: `--expert-cache 3000 --hot-ram-gib 21.0 --mmap-experts --pool-workers 12 --prefill 16384
--spec 6 --spec-min-p 0.8 --mtp mtp/rt --max-context 131072 --kv q4_0 --suffix-draft 3
--prompt-cache 6 --expert-cache-per-layer`, `STRATA_RSPLIT=1 STRATA_PREDICT=1`, port 8127.

Fresh reads (reused=0, engine-side `read in X ms` rate = client-side `prompt/TTFT` to 3 s.f.):

| prompt tok | read ms | prefill tok/s | decode tok/s | note |
| ---: | ---: | ---: | ---: | --- |
| 54,556 | 43,246 | 1,262 | — (0 gen) | best fresh rate in log |
| 33,260 | 31,826 | 1,045 | 43.2 | = bench `nvme-baseline-q2-32k.json` (1,041) |
| 4,206 | 5,862 | 718 | 50.1 | = bench `nvme-baseline-q2-quick.json` (713) |
| 48,464 | 69,623 | 696 | 29.3 | |
| 11,540 | 22,310 | 517 | 28.9 | |
| 1,099 | 4,368 | 252 | 40.2 | = bench `nvme-baseline-q2-quick.json` (248) |
| 121,633 | 111,646 | 1,089 | 30.4 | bench `nvme-q2-120k.json` (client TTFT view) |

Resume hits (conversation/prompt-cache: most tokens reused, TTFT collapses, so the
client-side quotient `prompt_tokens / TTFT` explodes — this is what the dashboard shows
in the 9–100k band; the engine only freshly reads the `fresh` column):

| prompt tok | reused | fresh | read ms | implied prefill tok/s | decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 27,092 | 27,074 | 18 | 457 | 59,282 | 30.1 |
| 27,911 | 27,900 | 11 | 559 | 49,930 | 21.7 |
| 63,806 | 63,774 | 32 | 1,449 | 44,035 | 20.1 |
| 65,133 | 65,073 | 60 | 1,487 | 43,802 | 28.7 |
| 41,909 | 41,869 | 40 | 968 | 43,294 | 10.3 |
| 105,262 | 105,202 | 60 | 2,486 | 42,342 | 21.8 |
| 34,997 | 34,967 | 30 | 849 | 41,221 | 27.0 |
| 11,459 | 11,454 | 5 | 332 | 34,515 | 29.0 |
| 106,409 | 106,297 | 112 | 3,196 | 33,294 | 25.6 |

Source: 385 `prompt … tokens = … reused + … read in … ms` rows in the live
`logs/sc117-q2-final.log` + `bench/results/nvme-baseline-q2-quick.json`,
`nvme-baseline-q2-32k.json`, `nvme-q2-120k.json`. Implied = `prompt / (read_ms/1000)`;
engine-side TTFT proxy (excludes queue/tokenizer), so client TTFT quotients land within ~5%.
Rule of thumb: fresh ≈ 250–1,260 tok/s by size; resumed ≈ prompt_tokens / ~0.3–3 s.
