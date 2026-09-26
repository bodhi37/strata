#!/usr/bin/env python3
"""probe_gguf.py - read a GGUF header (locally or over HTTP range) and report what
Strata's engine needs to know: the architecture guard keys, every ggml type present,
and the type of each routed-expert tensor.

    python3 probe_gguf.py <path-or-url> [--bytes N] [--json]

The engine's supported sets come from the source tree:
  CPU expert gate/up  : IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
  CPU expert down     : Q2_0, IQ4_NL
  dense mmvq          : Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS
  dense float         : BF16, F16, F32
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGML_TYPES  # noqa: E402

SIZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

ARCH_KEYS = (
    "general.architecture", "general.name", "general.size_label", "general.quantized_by",
    "qwen4exp.block_count", "qwen4exp.embedding_length", "qwen4exp.expert_count",
    "qwen4exp.expert_used_count", "qwen4exp.attention.head_count",
    "qwen4exp.attention.head_count_kv", "qwen4exp.context_length", "split.no", "split.count",
)

EXPERT_GU_OK = {"IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"}
EXPERT_DOWN_OK = {"Q2_0", "IQ4_NL"}
DENSE_OK = {"Q2_0", "Q4_0", "Q5_0", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS",
            "BF16", "F16", "F32"}


def fetch_head(src: str, nbytes: int) -> pathlib.Path:
    if not src.startswith(("http://", "https://")):
        return pathlib.Path(src)
    out = pathlib.Path(tempfile.gettempdir()) / ("probe-" + pathlib.Path(src).name)
    if out.exists() and out.stat().st_size >= nbytes // 2:
        return out
    subprocess.run(["curl", "-sL", "-r", f"0-{nbytes - 1}", src, "-o", str(out)], check=True)
    return out


def parse(path: pathlib.Path):
    d = path.read_bytes()
    if d[:4] != b"GGUF":
        head = d[:120].decode("utf-8", "replace")
        raise SystemExit(f"not a GGUF (starts with {head!r})")
    pos = 4
    ver, n_tensors, n_kv = struct.unpack_from("<IQQ", d, pos)
    pos += 20

    def rdstr(p):
        (n,) = struct.unpack_from("<Q", d, p)
        p += 8
        return d[p:p + n].decode("utf-8", "replace"), p + n

    meta: dict = {}
    for _ in range(n_kv):
        key, pos = rdstr(pos)
        (t,) = struct.unpack_from("<I", d, pos)
        pos += 4
        if t == 8:
            val, pos = rdstr(pos)
        elif t == 9:
            (et,) = struct.unpack_from("<I", d, pos)
            pos += 4
            (cnt,) = struct.unpack_from("<Q", d, pos)
            pos += 8
            if et == 8:
                val = f"ARRAY<String>[{cnt}]"
                for _ in range(cnt):
                    _, pos = rdstr(pos)
            else:
                val = f"ARRAY[{cnt}]"
                pos += SIZ[et] * cnt
        else:
            (val,) = struct.unpack_from("<" + FMT[t], d, pos)
            pos += SIZ[t]
        if key in ARCH_KEYS:
            meta[key] = val

    types = collections.Counter()
    experts = []
    for _ in range(n_tensors):
        name, pos = rdstr(pos)
        (nd,) = struct.unpack_from("<I", d, pos)
        pos += 4
        shape = struct.unpack_from(f"<{nd}Q", d, pos)
        pos += 8 * nd
        tid, _off = struct.unpack_from("<IQ", d, pos)
        pos += 12
        tname = GGML_TYPES.get(tid, f"type{tid}")
        types[tname] += 1
        if name.startswith("blk.") and name.endswith("_exps.weight"):
            experts.append((name, tname, tuple(shape)))
    return ver, n_tensors, n_kv, meta, types, experts, pos


def verdict(experts, types):
    gu, dn, bad = collections.Counter(), collections.Counter(), []
    for name, tname, _shape in experts:
        role = "gu" if ("ffn_gate_exps" in name or "ffn_up_exps" in name) else "down"
        (gu if role == "gu" else dn)[tname] += 1
    for t, c in gu.items():
        if t not in EXPERT_GU_OK:
            bad.append(f"gate/up {t} x{c}")
    for t, c in dn.items():
        if t not in EXPERT_DOWN_OK:
            bad.append(f"down {t} x{c}")
    dense_bad = sorted(t for t in types if t not in DENSE_OK and not t.startswith("type"))
    return gu, dn, bad, dense_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--bytes", type=int, default=12_000_000)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    path = fetch_head(a.src, a.bytes)
    ver, nt, nkv, meta, types, experts, hdr_end = parse(path)
    gu, dn, bad, dense_bad = verdict(experts, types)

    if a.json:
        print(json.dumps({"version": ver, "n_tensors": nt, "meta": meta,
                          "types": dict(types), "experts_gu": dict(gu), "experts_down": dict(dn),
                          "expert_incompatible": bad, "dense_incompatible": dense_bad}, indent=1))
        return 0

    print(f"{path}  GGUF v{ver}, {nt} tensors, {nkv} kv pairs, header ends at {hdr_end} B")
    for k in ARCH_KEYS:
        if k in meta:
            print(f"  {k} = {meta[k]}")
    print(f"  types: {dict(types)}")
    print(f"  expert gate/up: {dict(gu)}")
    print(f"  expert down   : {dict(dn)}")
    ok = meta.get("general.architecture") == "qwen4exp" and not bad and not dense_bad
    if bad:
        print(f"  EXPERT TYPES NOT SUPPORTED: {bad}")
    if dense_bad:
        print(f"  DENSE TYPES NOT SUPPORTED: {dense_bad}")
    print(f"  STRATA VERDICT: {'compatible' if ok else 'needs requantization'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
