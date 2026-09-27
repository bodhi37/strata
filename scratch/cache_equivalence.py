#!/usr/bin/env python3
"""plan v0.4 P8c: protocol-level proof that cache reuse is state-exact.

Drives `engine/strata --serve` directly over its stdin/stdout line protocol and proves the
hard property: the tokens a REUSED session produces for a continuation request are identical,
token-for-token, to what a from-scratch full prefill of the same stream produces.

    python3 scratch/cache_equivalence.py [--engine engine/strata] [--serve-args ...]

Steps:
  1. GEN  P1                     -> R1 (turn 1, full prefill)
  2. GEN  KEEP  P1+R1+S2         -> O2 (turn 2 via the cache; assert reused == len(P1)+len(R1))
  3. GEN  P0 (disjoint prompt)   -> force a miss, wiping the session
  4. GEN  P1+R1+S2 (no KEEP)     -> O2' (same stream, full prefill from zero)
  5. assert O2 == O2' exactly; assert every turn's DONE counters
"""
import argparse
import random
import subprocess
import sys
import time

IM_END = 151645  # <|im_end|>


class Engine:
    def __init__(self, cmd, serve_args, cfg=None):
        import os
        env = dict(os.environ)
        dirs = [d for d in (cfg or {}).get("lib_dirs") or [] if os.path.isdir(d)]
        if dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs +
                                                     ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        env.update({str(k): str(v) for k, v in ((cfg or {}).get("env") or {}).items()})
        self.p = subprocess.Popen([*cmd, "--serve", *serve_args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=sys.stderr, text=True, encoding="utf-8", bufsize=1, env=env)
        line = self.p.stdout.readline()
        if not line.startswith("READY"):
            raise RuntimeError("engine not ready: " + line.strip())
        self.max_context = int(line.split()[1])

    def gen(self, max_new, ids, keep=None):
        head = f"GEN {max_new}" if keep is None else f"GEN {max_new} KEEP {keep}"
        self.p.stdin.write(f"{head} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        out, done = [], None
        for line in self.p.stdout:
            if line.startswith("T "):
                out.append(int(line[2:]))
            elif line.startswith("DONE"):
                f = line.split()
                done = {"generated": int(f[1]), "prompt": int(f[2]), "finish": f[5], "sess": int(f[6]),
                        "reused": int(f[7]) if len(f) > 7 else -1, "prefilled": int(f[8]) if len(f) > 8 else -1}
                break
            elif line.startswith("ERR"):
                raise RuntimeError("engine error: " + line[4:].strip())
        return out, done

    def quit(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=15)
        except Exception:
            self.p.kill()


def build_prompts(tok_dir, n2_text="  The reef itself was older than the charts claimed; storms had redrawn it twice."):
    """Natural-text prompt segments (near-uniform logits on random ids turn argmax into coin
    flips; real text gives confident logits so only real state divergence can flip a token)."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_dir.rstrip("/") + "/tokenizer.json")
    passage = ("The lighthouse keeper Elara kept a ledger of every ship that passed the reef. In forty years "
               "she had logged 12,304 vessels and saved 97 crews, and she never once left the lamp unlit. "
               "Her cat Barnaby supervised from the windowsill each evening.")
    p1 = tk.encode(passage).ids
    s2 = tk.encode(n2_text).ids
    rng = random.Random(7)
    disjoint = tk.encode("Completely unrelated filler about quantum kettles and municipal bicycle policy. "
                         "The council met on Tuesday and voted nine to four against the proposal. ").ids
    disjoint = (disjoint * 2)[:96]
    return p1, s2, disjoint


def serve_args_from_config(path):
    """The exact spawn serve/server.py does: the config's exe (an mlock/nice wrapper, when set)
    plus its args (and cfg, for the CUDA library path)."""
    import json
    cfg = json.load(open(path))
    return [cfg["exe"]] if cfg.get("exe") else [], list(cfg["args"]), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None,
                    help="engine binary; default: the config's exe (run-engine-mlock.sh)")
    ap.add_argument("--config", default="strata-r12-static.json",
                    help="an r*-*.json engine config; its exe+args spawn `--serve`")
    ap.add_argument("--serve-args", nargs="*", default=None, help="override the config's engine args")
    ap.add_argument("--max-new", type=int, default=24)
    a = ap.parse_args()
    exe, args, cfg = serve_args_from_config(a.config)
    cmd = [a.engine] if a.engine else exe
    eng = Engine(cmd, args, cfg)
    tok_dir = cfg.get("tokenizer") or (a.config and "packs/heretic-iq3xxs/tokenizer")
    p1, s2, disjoint = build_prompts(tok_dir)
    ok = True
    try:
        # ---- turn 1: full prefill of P1, response R1
        r1, d1 = eng.gen(a.max_new, p1)
        print(f"turn1: prompt {d1['prompt']} gen {d1['generated']} sess {d1['sess']} "
              f"reused {d1['reused']} prefilled {d1['prefilled']} finish {d1['finish']}")
        # a stop finish commits the in-flight stop token via the fix-up window (prefilled = n);
        # a length cut mid-window commits exactly the emitted stream (prefilled = n-1).  Both
        # leave sess == prompt + emitted exactly.
        assert d1["reused"] == 0 and d1["prefilled"] in (len(p1), len(p1) - 1), "turn 1 is a full prefill"
        assert d1["sess"] == len(p1) + len(r1), f"sess {d1['sess']} != prompt+emitted {len(p1)+len(r1)}"
        assert r1 and d1["finish"] in ("stop", "length"), "turn 1 should end cleanly"

        # ---- turn 2 through the cache: the record (P1+R1) extended by S2
        turn2 = p1 + r1 + s2
        o2, d2 = eng.gen(a.max_new, turn2, keep=len(p1) + len(r1))
        print(f"turn2: prompt {d2['prompt']} gen {d2['generated']} sess {d2['sess']} "
              f"reused {d2['reused']} prefilled {d2['prefilled']} finish {d2['finish']}")
        assert d2["reused"] == len(p1) + len(r1), \
            f"cache did not engage: reused {d2['reused']} != {len(p1)+len(r1)}"
        assert d2["prefilled"] == len(s2), f"turn 2 must prefill only the new suffix, got {d2['prefilled']}"
        assert d2["sess"] == len(turn2) + len(o2)

        # ---- force a miss with a disjoint prompt (content mismatch -> full reset)
        _r0, d0 = eng.gen(4, disjoint)
        print(f"miss:  reused {d0['reused']} (must be 0)")
        assert d0["reused"] == 0, "a disjoint prompt must be a miss"

        # ---- the same turn-2 stream, full prefill from zero
        o2p, d2p = eng.gen(a.max_new, turn2)
        print(f"full:  prompt {d2p['prompt']} gen {d2p['generated']} reused {d2p['reused']} "
              f"prefilled {d2p['prefilled']} finish {d2p['finish']}")
        assert d2p["reused"] == 0, "the full-prefill control must not reuse"

        # ---- THE PROPERTY: identical outputs for the same canonical token stream
        if o2 != o2p:
            first = next((i for i, (x, y) in enumerate(zip(o2, o2p)) if x != y), None)
            print(f"MISMATCH at token {first}:\n  reused  {o2}\n  prefill {o2p}")
            ok = False
        else:
            print(f"EQUIVALENCE OK: {len(o2)} tokens identical between reused and full-prefill turns")

        # ---- determinism control: the same full prefill AGAIN on the same path.  If this also
        # diverges, the engine itself is not deterministic and the comparison above is void.
        o2q, d2q = eng.gen(a.max_new, turn2)
        if o2p != o2q:
            print(f"DETERMINISM CONTROL FAILED:\n  control1 {o2p}\n  control2 {o2q}")
            ok = False
        else:
            print("determinism control OK: two identical full prefills agree exactly")

        # ---- turn 3 chain: the reused session keeps extending
        turn3 = turn2 + o2p + s2
        r3, d3 = eng.gen(a.max_new, turn3, keep=len(turn2) + len(o2p))
        print(f"turn3: reused {d3['reused']} prefilled {d3['prefilled']}")
        assert d3["reused"] == len(turn2) + len(o2p), "turn 3 must chain on turn 2's cache"
        assert d3["prefilled"] == len(s2), "turn 3 must prefill only the new suffix"
    finally:
        eng.quit()
    if not ok:
        sys.exit(1)
    print("cache equivalence: PASS")


if __name__ == "__main__":
    main()
