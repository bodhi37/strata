#!/usr/bin/env python3
"""tools/append_expert_tensor.py - fix a layer whose expert tensors straddle two GGUF shards.

Strata's native_experts.txt names ONE file per layer, and load_experts_gguf reads the raw bytes at
absolute file offsets - so a layer whose ffn_*_exps tensors were split across shards 4/5 by the
uploader's split point can be served by APPENDING the out-of-place tensor's raw bytes to the shard
that holds the other two.  The GGUF headers are untouched (parsers ignore trailing bytes), and the
recorded absolute offset goes into a small JSON that tools/iq_pack.py consumes via
--layer-experts-override.

Usage:
  venv/bin/python tools/append_expert_tensor.py --from model/A-00004-of-00005.gguf \
      --to model/A-00005-of-00005.gguf --tensor blk.36.ffn_down_exps.weight \
      --layer 36 --out model/layer-experts-override-IQ3_M.json
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as gr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-shard", required=True)
    ap.add_argument("--to-shard", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = gr.GGUFFile(a.from_shard)
    t = next((t for t in src.tensors if t.name == a.tensor), None)
    if t is None:
        print("ERROR: %s not in %s" % (a.tensor, a.from_shard))
        return 1
    n = t.expected_bytes()
    start = src.data_start + t.offset

    dst_path = pathlib.Path(a.to_shard)
    dst_size = dst_path.stat().st_size
    copied = 0
    with open(a.from_shard, "rb") as fi, open(a.to_shard, "ab") as fo:
        fi.seek(start)
        while copied < n:
            chunk = fi.read(min(8 << 20, n - copied))
            if not chunk:
                print("ERROR: short read copying %s" % a.tensor)
                return 1
            fo.write(chunk)
            copied += len(chunk)
    print("appended %s (%d B) to %s at absolute offset %d" % (a.tensor, copied, a.to_shard, dst_size))

    # the override JSON: final absolute offsets in the destination shard for all three roles
    dst = gr.GGUFFile(a.to_shard)
    offs = {}
    for role in ("gate", "up", "down"):
        name = "blk.%d.ffn_%s_exps.weight" % (a.layer, role)
        tt = next((x for x in dst.tensors if x.name == name), None)
        offs[role] = (dst.data_start + tt.offset) if tt is not None else None
    if offs["down"] is None:      # this is the appended one
        offs["down"] = dst_size
    # any OTHER role still absent from dst lives in the source shard and must be moved too:
    missing = [r for r in ("gate", "up") if offs[r] is None]
    if missing:
        print("ERROR: roles %s are not in the destination shard either - move them first" % missing)
        return 1
    obj = {str(a.layer): {"shard": dst_path.name, "gate": offs["gate"], "up": offs["up"], "down": offs["down"]}}
    pathlib.Path(a.out).write_text(json.dumps(obj, indent=1))
    print("wrote %s: %s" % (a.out, json.dumps(obj)))
    return 0


if __name__ == "__main__":
    sys.exit(main())