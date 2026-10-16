#!/usr/bin/env python3
"""sim.py - offline residency simulator over a STRATA_ROUTE_TRACE file.

Replays every verify window (union of routed experts per layer) against:
  * a VRAM tier  : top of the ranking, per-layer quota (the engine's R4.2g per-layer slots) or global
  * a RAM tier   : next slice of the ranking, packed by actual blob bytes (4 KiB aligned)
  * an LRU       : optional fixed-slot recency cache that admits every miss (zero-copy ring admit)
and reports missed blobs / bytes per ACCEPTED token - the number decode speed is inversely proportional to
when the drive is the wall.

  hill/sim.py TRACE --counts data/counts.strc --vram-slots 1500 --ram-gib 24 [--lru-gib 2] [--per-layer]
"""
from __future__ import annotations
import argparse, os, struct, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL, NE = 48, 512


def blob_bytes():
    bb = np.zeros(NL, dtype=np.int64)
    for line in open(os.path.join(REPO, "packs/sc117-iq3s/native_experts.txt")):
        if line.startswith("#"):
            continue
        f = line.split()
        bb[int(f[0])] = int(f[4])
    return bb


def load_trace(path):
    """-> list of windows: (T, accepted, sfx, [per-layer np.array of ids (n_tok*k)])"""
    raw = np.fromfile(path, dtype="<i2")
    wins, cur, i, n = [], {}, 0, len(raw)
    while i + 3 <= n:
        layer, nt, k = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
        i += 3
        if layer == -1:
            acc, sfx = (k - 1000, True) if k >= 1000 else (k, False)
            if cur:
                wins.append((nt, acc, sfx, cur))
            cur = {}
            continue
        if not (0 <= layer < NL) or nt <= 0 or k <= 0 or i + nt * k > n:
            break
        if layer == 0 and cur:          # no footer: a prompt-reading window, not decode - dropped
            cur = {}
        cur[layer] = raw[i:i + nt * k].copy()
        i += nt * k
    return wins


def load_counts(path):
    b = open(path, "rb").read()
    nl, ne, _ = struct.unpack("<3I", b[4:16])
    return np.frombuffer(b, "<u8", nl * ne, 16).astype(np.float64)


def ranking_from_counts(c, bb=None):
    """by count, or (bb given) by count per byte - the greedy knapsack order for a byte-budgeted tier"""
    key = c if bb is None else c / np.repeat(bb, NE)
    return [int(i) for i in np.argsort(-key, kind="stable") if c[i] > 0]


def build_tiers(rank, bb, vram_slots, ram_bytes, per_layer):
    vram = np.zeros(NL * NE, bool)
    if vram_slots > 0:
        if per_layer:
            q = vram_slots // NL
            per = np.zeros(NL, int)
            for idx in rank:
                l = idx // NE
                if per[l] < q:
                    vram[idx] = True
                    per[l] += 1
        else:
            for idx in rank[:vram_slots]:
                vram[idx] = True
    ram = np.zeros(NL * NE, bool)
    used = 0
    for idx in rank:
        if vram[idx]:
            continue
        p = (int(bb[idx // NE]) + 4095) // 4096 * 4096
        if used + p > ram_bytes:
            break
        ram[idx] = True
        used += p
    return vram, ram


def simulate(wins, vram, ram, bb, lru_slots=0, skip_rejected=False, max_t=0, base_ms=14.0, lat_ms=0.45,
             bw_gbs=4.2):
    """time model per window: base_ms (GPU + sync) + per layer with >=1 miss: lat_ms + bytes/bw"""
    from collections import OrderedDict
    lru = OrderedDict()
    miss_blobs = miss_bytes = 0
    acc_tok = 0
    routed = 0
    t_ms = 0.0
    for (T, acc, sfx, layers) in wins:
        if max_t and T > max_t:
            T2 = max_t
            acc = min(acc, max_t - 1) if acc >= 0 else acc
        else:
            T2 = T
        acc_tok += (acc + 1) if acc >= 0 else 1
        t_ms += base_ms
        for l, ids in layers.items():
            nt = T if T > 0 else 1
            k = len(ids) // max(nt, 1)
            ids = ids[:max(T2, 1) * k]
            if skip_rejected and acc >= 0 and T > 0:
                ids = ids[:(acc + 1) * k]
            u = np.unique(ids + l * NE)
            lb = 0
            routed += len(ids)
            for idx in u:
                if vram[idx] or ram[idx]:
                    continue
                if lru_slots:
                    if idx in lru:
                        lru.move_to_end(idx)
                        continue
                    lru[idx] = 1
                    if len(lru) > lru_slots:
                        lru.popitem(last=False)
                miss_blobs += 1
                miss_bytes += int(bb[l])
                lb += int(bb[l])
            if lb:
                t_ms += lat_ms + lb / (bw_gbs * 1e6)
    return dict(windows=len(wins), acc_tok=acc_tok, miss_per_tok=miss_blobs / acc_tok,
                mb_per_tok=miss_bytes / acc_tok / 1e6, tok_per_win=acc_tok / max(1, len(wins)),
                model_tps=acc_tok / (t_ms / 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--counts", required=True, help="STRC counts for the ranking (comma list = summed)")
    ap.add_argument("--vram-slots", type=int, default=1500)
    ap.add_argument("--ram-gib", type=float, default=24.0)
    ap.add_argument("--lru-gib", type=float, default=0.0)
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--skip-rejected", action="store_true")
    ap.add_argument("--density", action="store_true", help="rank by count/byte")
    ap.add_argument("--max-t", type=int, default=0)
    a = ap.parse_args()
    bb = blob_bytes()
    if a.counts.endswith(".bin"):           # an STRP profile: the rank list itself
        b = open(a.counts, "rb").read()
        _, nl, ne, slots, nr = struct.unpack("<5I", b[4:24])
        pr = np.frombuffer(b, "<u2", 2 * nr, 24).reshape(-1, 2)
        rank = [int(l) * NE + int(e) for l, e in pr]
    else:
        c = sum(load_counts(p) for p in a.counts.split(","))
        rank = ranking_from_counts(c, bb if a.density else None)
    wins = load_trace(a.trace)
    lru_slots = int(a.lru_gib * 2**30 // (2662400 + 4096)) if a.lru_gib else 0
    ram_bytes = int((a.ram_gib - a.lru_gib) * 2**30)
    vram, ram = build_tiers(rank, bb, a.vram_slots, ram_bytes, a.per_layer)
    r = simulate(wins, vram, ram, bb, lru_slots, a.skip_rejected, a.max_t)
    print(f"vram {vram.sum()} ram {ram.sum()} lru {lru_slots} | windows {r['windows']} acc_tok {r['acc_tok']} "
          f"tok/win {r['tok_per_win']:.2f} | miss/tok {r['miss_per_tok']:.1f}  MB/tok {r['mb_per_tok']:.1f}  "
          f"-> disk ceiling {4200 / max(r['mb_per_tok'], 1e-9):.0f} tok/s, model {r['model_tps']:.1f} tok/s")


if __name__ == "__main__":
    main()
