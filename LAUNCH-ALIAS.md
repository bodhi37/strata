# Launch aliases — Qwen3.8-Flash-Next (Strata)

One engine at a time on this box (30 GiB RAM: the 24.5 GiB mlocked tier + ~4 GiB engine = all of it).
Two engines = OOM. Ports: 8111 is the default and is ALSO used by the kv-cache agent's harness —
if that agent is active, launch on 8123+ to avoid being killed mid-run.

## Start / stop

```
cd ~/models/strata
bash srv.sh <config.json> [port]        # start (tmux, waits for readiness); default port 8111
bash srv.sh restart <config.json> [port]
bash srv.sh stop                        # stop whatever is on the tmux session for that port
```

Manual (no tmux): `./venv/bin/python serve/server.py --engine strata --config <cfg> --port <p> &`

## Quants

| alias | quant | config | pack | notes |
| --- | --- | --- | --- | --- |
| `qwen3.8-flash-iq3` / `-iq3xxs` | heretic-2 IQ3_XXS | `strata-r13-base.json` | `packs/heretic-iq3xxs` | 49.8 GiB arena, best-fit for 30 GiB RAM; measured median ~8 tok/s @200tok, 14.1 @300tok (r13-base) |
| `qwen3.8-flash-iq3m` | heretic-2 IQ3_M | `strata-iq3_m.json` (r7 lineage) | `packs/heretic-iq3_m` | IQ3_S gate/up + Q5_0 down x6 layers (kernels patched) |
| `qwen3.8-flash-iq4` / `-iq4xs` | heretic-2 IQ4_XS | `strata-iq4_xs-v7.json` | `packs/heretic-iq4_xs` | smartest (KLD 0.0472), 65.4 GiB arena — tightest fit |

Aliases are config names; launch with the table above (`srv.sh strata-r13-base.json 8123` etc).
Config lineage `strata-r7-* … strata-r13-*` records every tried knob — read before repeating.

## Critical environment facts (learned the hard way)

1. **GPU persistence mode MUST be on** or engine startup fails with
   `native embedding: cannot pin 644 MiB (cuda: CUDA-capable device(s) is/are busy or unavailable)`.
   Fix: `echo 'Monster.8!!!' | sudo -S nvidia-smi -pm 1` (resets on reboot).
2. After killing an engine, wait ~10 s before starting a new one — the driver needs to clean up
   the killed context (NV_ERR_STATE_IN_USE errors otherwise).
3. `pkill -f 'serve/server.py'` kills YOUR OWN shell if the pattern appears in your command line
   (srv.sh uses `[e]` bracket classes for this reason). Kill by PID or use srv.sh.
4. One engine only. Check `ps aux | grep engine/strata` before launching.
5. First run of a session: give the engine ~90 s to load (PLE 28.8 GB + weights + tier fill).

## Current best configs (IQ3_XXS)

- `strata-r13-base.json` — static 24.5 GiB tier from `data/profile-r10.bin`, O_DIRECT reads,
  spec 4, min-p 0.8, kv int8, 131072 ctx, port 8111/8123.
- `strata-r13-admit.json` — dynamic LRU tier + prefill admissions ON (multi-turn/agentic reuse;
  UNTESTED as of handoff — bench was interrupted).
- `strata-r13-spec6.json` (admit + spec 6), `strata-r13-spin.json` (+ park spin 250) — staged A/B arms.

## Pi harness

Endpoint `http://127.0.0.1:<port>/v1` (OpenAI + Anthropic compatible); model name
`qwen3.8-flash-next-heretic-2-iq3_xxs`. Context 131072, reasoning via `reasoning_effort`.
