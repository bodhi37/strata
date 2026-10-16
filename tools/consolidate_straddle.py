#!/usr/bin/env python3
"""tools/consolidate_straddle.py - make every layer's expert tensors servable from one shard.

A quant's 5-shard split point can fall INSIDE a layer's ffn_*_exps set (e.g. IQ3_M layer 36:
gate/up in shard 5, down in shard 4).  Strata's native_experts.txt names one file per layer, so the
out-of-place tensors are APPENDED (raw bytes, headers untouched) to the shard that already holds
the majority of the layer's experts, and the absolute offsets go into the JSON that iq_pack.py
consumes via --layer-experts-override.  Idempotent: layers already in one shard are left alone.

Usage:
  venv/bin/python tools/consolidate_straddle.py \
      --model model/Qwen3.8-Flash-Next-heretic-2-IQ3_M-00001-of-00005.gguf \
      --out model/layer-experts-override-IQ3_M.json
"""
import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as gr

ROLES = ("gate", "up", "down")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="first shard; siblings auto-discovered")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    first = pathlib.Path(a.model)
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if not m:
        print("ERROR: --model must be shard 1 of a -00001-of-00005 split")
        return 1
    total = int(m.group(2))
    shards = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
              for i in range(1, total + 1)]
    files, where = [], {}
    for p in shards:
        if not p.exists():
            continue
        f = gr.GGUFFile(str(p))
        files.append((p, f))
        for t in f.tensors:
            if t.name.startswith("blk.") and t.name.endswith("_exps.weight"):
                where[t.name] = (p, f, t)

    # group the expert tensors by layer
    by_layer = collections.defaultdict(dict)
    for name, (p, f, t) in where.items():
        mm = re.match(r"blk\.(\d+)\.ffn_(\w+)_exps\.weight$", name)
        if mm and mm.group(2) in ROLES:
            by_layer[int(mm.group(1))][mm.group(2)] = (p, f, t)

    override, fixed = {}, []
    for layer in sorted(by_layer):
        roles = by_layer[layer]
        if len(roles) != 3:
            print("ERROR: layer %d has only %s of its expert tensors in the model" % (layer, sorted(roles)))
            return 1
        shards_of = {r: roles[r][0] for r in ROLES}
        if len(set(shards_of.values())) == 1:
            continue                     # nothing to do
        # majority shard wins; append the minorities to it
        counts = collections.Counter(shards_of.values())
        dst_path = counts.most_common(1)[0][0]
        dst = gr.GGUFFile(str(dst_path))
        dst_size = dst_path.stat().st_size
        offs = {}
        with open(dst_path, "ab") as fo:
            for r in ROLES:
                p, f, t = roles[r]
                if p == dst_path:
                    offs[r] = dst.data_start + t.offset
                    continue
                n = t.expected_bytes()
                copied = 0
                with open(p, "rb") as fi:
                    fi.seek(f.data_start + t.offset)
                    while copied < n:
                        chunk = fi.read(min(8 << 20, n - copied))
                        if not chunk:
                            print("ERROR: short read copying %s" % t.name)
                            return 1
                        fo.write(chunk)
                        copied += len(chunk)
                offs[r] = dst_size
                dst_size += copied
                print("layer %d: appended %s (%d B) to %s @ %d" % (layer, t.name, copied, dst_path.name, dst_size - copied))
        override[str(layer)] = {"shard": dst_path.name, **offs}
        fixed.append(layer)

    pathlib.Path(a.out).write_text(json.dumps(override, indent=1) if override else "{}")
    if fixed:
        print("consolidated layers %s; wrote %s" % (fixed, a.out))
    else:
        print("no straddling layers; wrote empty %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())