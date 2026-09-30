#!/usr/bin/env python3
"""tools/make_native_head.py - a one-tensor GGUF holding `output.weight`, plus the architecture keys.

WHY THIS EXISTS.  `NativeHead::load` (src/core/native_head.cpp) opens ONE file and calls
`check_architecture` on it before it looks for `output.weight`.  A split community quant can put
the head in a different shard from the metadata: Qwen3.8-Flash-Next-heretic-2-i1 keeps the 68
architecture keys in shard 1 (0 tensors) and `output.weight` in shard 2 (3 keys), so neither file
satisfies both conditions.  This tool writes a tiny sidecar with the guard keys copied from the
metadata shard and the head tensor copied byte-for-byte from the tensor shard, which is what
`--native-head-gguf` wants.  The trunk still loads normally from the real shards.

    python3 tools/make_native_head.py --arch-shard ...-00001-of-00005.gguf \
        --tensor-shard ...-00002-of-00005.gguf --out head.gguf
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

GUARD = ("general.architecture", "qwen4exp.block_count", "qwen4exp.embedding_length",
         "qwen4exp.expert_count", "qwen4exp.expert_used_count", "qwen4exp.attention.head_count",
         "qwen4exp.attention.head_count_kv")


def wr_u32(f, v): f.write(struct.pack("<I", v))
def wr_u64(f, v): f.write(struct.pack("<Q", v))
def wr_str(f, s):
    b = s.encode()
    wr_u64(f, len(b)); f.write(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch-shard", required=True)
    ap.add_argument("--tensor-shard", required=True)
    ap.add_argument("--name", default="output.weight")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arch = G.GGUFFile(a.arch_shard)
    tsh = G.GGUFFile(a.tensor_shard)
    meta = arch.metadata
    missing = [k for k in GUARD if k not in meta]
    if missing:
        print("architecture shard is missing: %s" % ", ".join(missing)); return 1
    t = next((x for x in tsh.tensors if x.name == a.name), None)
    if t is None:
        print("%s has no tensor %s" % (a.tensor_shard, a.name)); return 1
    if len(t.shape) != 2:
        print("%s is not 2-D" % a.name); return 1
    # Stream the tensor bytes with seek + chunked copy: the source shard can be
    # tens of GB and must never be read fully into RAM (OOM).
    src_off = tsh.data_start + t.offset
    nbytes = t.expected_bytes()
    if nbytes is None or nbytes <= 0:
        print("cannot determine byte size of %s" % a.name); return 1

    out = pathlib.Path(a.out)
    with out.open("wb") as f:
        wr_u32(f, 0x46554747); wr_u32(f, 3); wr_u64(f, 1); wr_u64(f, len(GUARD) + 1)
        wr_str(f, "general.alignment"); wr_u32(f, 4); wr_u32(f, 4096)
        for k in GUARD:
            v = meta[k]
            wr_str(f, k)
            if isinstance(v, str):
                wr_u32(f, 8); wr_str(f, v)
            else:
                wr_u32(f, 4); wr_u32(f, int(v))
        wr_str(f, a.name); wr_u32(f, 2)
        wr_u64(f, int(t.shape[0])); wr_u64(f, int(t.shape[1]))
        wr_u32(f, t.type_id); wr_u64(f, 0)
        hdr = f.tell()
        f.write(b"\0" * ((-hdr) % 4096))
        data_start = f.tell()
        with open(a.tensor_shard, "rb") as src:
            src.seek(src_off)
            remaining = nbytes
            while remaining > 0:
                chunk = src.read(min(64 * 1024 * 1024, remaining))
                if not chunk:
                    print("short read of %s" % a.name); return 1
                f.write(chunk)
                remaining -= len(chunk)
    expect = data_start + nbytes
    print("%s: %s %s %s -> %d bytes (expected %d)" % (out, a.name, t.type_name, list(t.shape),
                                                      out.stat().st_size, expect))
    return 0 if out.stat().st_size == expect else 1


if __name__ == "__main__":
    raise SystemExit(main())
