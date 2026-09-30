#!/usr/bin/env python3
"""tools/repair_heretic_pack.py - fix a heretic-2 IQ pack for Strata's canonical-contract tensors.

The heretic-2 quantizer stored EVERY 2-D tensor as Q8_0, including the small per-layer tensors the
Strata engine consumes in canonical BF16 form (readers cast the arena data to `const uint16_t*`):

  * blk.N.hc_attn_down/up/inject.weight, blk.N.hc_ffn_down/up/inject.weight  (48 layers x 6)
  * blk.N.ssm_alpha.weight, blk.N.ssm_beta.weight                            (36 GDN layers x 2)
  * output_hc_down.weight, output_hc_up.weight
  * blk.1.ple_value.weight                                                   (PLE value matrix)

tools/iq_pack.py marks quantized tensors "native-only" (dst_bytes 0), which the engine rejects for
these names (they are not in NativeDense's mmvq whitelist, and gr_read/project_bf16/fused_gdn_ab
want raw BF16). This script dequantizes Q8_0 -> BF16 (round-to-nearest-even) directly from the model
shards and rewrites the index rows as canonical kind-4 (raw BF16, 2 B/elem) rows appended to dense.bin.

Idempotent: rows already carrying kind 4 with dst_bytes > 0 are left alone.

Usage: venv/bin/python tools/repair_heretic_pack.py --pack packs/heretic-iq3xxs \
          --model model/Qwen3.8-Flash-Next-heretic-2-IQ3_XXS-00001-of-00005.gguf
"""
import argparse
import numpy as np
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as gr

ALIGN = 256
REPAIR_SUFFIXES = (".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                   ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight",
                   ".ssm_alpha.weight", ".ssm_beta.weight")
REPAIR_NAMES = {"output_hc_down.weight", "output_hc_up.weight", "blk.1.ple_value.weight"}


def repair_set(name: str) -> bool:
    return name in REPAIR_NAMES or (name.startswith("blk.") and name.endswith(REPAIR_SUFFIXES))


def dequant_q8_0_to_bf16(raw: bytes) -> bytes:
    n = len(raw) // 34
    u = np.frombuffer(raw, dtype=np.uint8, count=n * 34).reshape(n, 34)
    d = u[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)  # [n]
    qs = u[:, 2:].copy().view(np.int8).astype(np.float32)                # [n, 32]
    x = (d[:, None] * qs).reshape(-1)                                # f32, row-major ne0-fast
    # f32 -> bf16, round to nearest even: add 0x7FFF + lsb carry, keep the high half
    b = x.view(np.uint32).copy()
    b += np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))
    return (b >> np.uint32(16)).astype(np.uint16).tobytes()


# ggml's IQ4_NL codebook (ggml-quants.c: kvalues_iq4nl); the CUDA kernel
# vec_dot_iq4_xs_q8_1 and dq_iq4_xs decode against the same 16 numbers.
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def _f32_to_bf16_rne(x: np.ndarray) -> bytes:
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).copy()
    b += np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))
    return (b >> np.uint32(16)).astype(np.uint16).tobytes()


def dequant_iq4_nl_to_bf16(raw: bytes) -> bytes:
    """IQ4_NL blocks (32 vals, 18 B: fp16 d + 16 B nibbles) -> BF16 bytes."""
    n = len(raw) // 18
    u = np.frombuffer(raw, dtype=np.uint8, count=n * 18).reshape(n, 18)
    d = u[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    qs = u[:, 2:]
    lo = KVALUES_IQ4NL[qs & 0x0F]
    hi = KVALUES_IQ4NL[qs >> 4]
    x = np.empty((n, 32), dtype=np.float32)
    x[:, :16] = d[:, None] * lo
    x[:, 16:] = d[:, None] * hi
    return _f32_to_bf16_rne(x.reshape(-1))


def dequant_iq4_xs_to_bf16(raw: bytes) -> bytes:
    """IQ4_XS super-blocks (256 vals, 136 B) -> BF16 bytes.

    Layout (ggml-common.h block_iq4_xs, QK_K=256): fp16 d, u16 scales_h,
    4 x u8 scales_l, 128 B nibbles.  Reference: ggml dequantize_row_iq4_xs
    and this repo's dq_iq4_xs (src/kernels/cuda/iq_kernels.cu).
    """
    n = len(raw) // 136
    u = np.frombuffer(raw, dtype=np.uint8, count=n * 136).reshape(n, 136)
    d = u[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)          # (B,)
    scales_h = u[:, 2:4].copy().view("<u2").reshape(-1)                    # (B,)
    scales_l = u[:, 4:8].astype(np.uint16)                                 # (B,4)
    qs = u[:, 8:].reshape(n, 8, 16)                                        # (B,8,16)
    ib = np.arange(8, dtype=np.uint16)
    ls = ((scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | \
         (((scales_h[:, None] >> (2 * ib)) & 3) << 4)                       # (B,8)
    dl = d[:, None] * (ls.astype(np.float32) - 32.0)                       # (B,8)
    lo = KVALUES_IQ4NL[qs & 0x0F]
    hi = KVALUES_IQ4NL[qs >> 4]
    x = np.empty((n, 8, 32), dtype=np.float32)
    x[:, :, :16] = dl[:, :, None] * lo
    x[:, :, 16:] = dl[:, :, None] * hi
    return _f32_to_bf16_rne(x.reshape(-1))


REPAIR_DEQUANT = {
    "Q8_0": dequant_q8_0_to_bf16,
    "IQ4_NL": dequant_iq4_nl_to_bf16,
    "IQ4_XS": dequant_iq4_xs_to_bf16,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model", required=True, help="first shard; siblings auto-discovered")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pack = pathlib.Path(a.pack)

    # ---- locate every repair tensor in the model shards
    first = pathlib.Path(a.model)
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    shards = [first] if not m else [
        first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, int(m.group(2))))
        for i in range(1, int(m.group(2)) + 1)]
    where = {}
    for sh in shards:
        f = gr.GGUFFile(str(sh))
        mm = np.memmap(str(sh), dtype=np.uint8, mode="r")
        for t in f.tensors:
            if repair_set(t.name):
                where[t.name] = (f, t, mm)
    print("model: %d shards, %d repair tensors located" % (len(shards), len(where)))

    # ---- parse the index
    lines = (pack / "index.txt").read_text().splitlines()
    align = 256
    for l in lines:
        if l.startswith("# align "):
            align = int(l.split()[2])
            break
    rows = [l.split() for l in lines if l.strip() and not l.startswith("#")]
    todo = [r for r in rows if repair_set(r[0]) and not (r[2] == "4" and int(r[6]) > 0)]
    already = sum(1 for r in rows if repair_set(r[0]) and r[2] == "4" and int(r[6]) > 0)
    print("index: %d rows, %d to repair, %d already canonical" % (len(rows), len(todo), already))
    missing = {r[0] for r in todo} - set(where)
    if missing:
        print("ERROR: repair tensors absent from the model shards: %s" % sorted(missing))
        return 1
    names_todo = {r[0] for r in todo}

    dense = pack / "dense.bin"
    at = dense.stat().st_size
    print("dense.bin: %d bytes now" % at)

    # ---- dequantize + append, rewrite rows in place
    new_rows = []
    with open(dense, "ab") as fo:
        for r in rows:
            if r[0] not in names_todo:
                new_rows.append(r)
                continue
            f, t, mm = where[r[0]]
            dq = REPAIR_DEQUANT.get(t.type_name)
            if dq is None:
                print("ERROR: %s is %s, no dequant path (have %s)" %
                      (t.name, t.type_name, sorted(REPAIR_DEQUANT)))
                return 1
            ne0, ne1 = int(t.shape[0]), int(t.shape[1] if len(t.shape) > 1 else 0)
            n = t.expected_bytes()
            raw = bytes(mm[f.data_start + t.offset: f.data_start + t.offset + n])
            bf16 = dq(raw)
            elems = ne0 * (ne1 if ne1 else 1)
            if len(bf16) != elems * 2:
                print("ERROR: %s dequant size %d != %d" % (t.name, len(bf16), elems * 2))
                return 1
            if not a.dry_run:
                fo.write(bf16)
                pad = (-len(bf16)) % align
                if pad:
                    fo.write(b"\0" * pad)
            # name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits ... (19 fields)
            nr = [r[0], "0", "4", str(at), str(len(bf16)), "0", str(len(bf16)), str(ne0), str(ne1),
                  "0", "0", "1", "0", "0", "0", "0", "0", "0", "0"]
            at += len(bf16) + ((-len(bf16)) % align)
            new_rows.append(nr)
            print("  %s: %s [%d, %d] -> BF16 %d B @ dense.bin %s" %
                  (r[0], t.type_name, ne0, ne1, len(bf16), nr[3]))

    if a.dry_run:
        print("dry run: index not rewritten")
        return 0

    # ---- recompute dst_off column + header pool
    dst = 0
    for r in new_rows:
        r[5] = str(dst)
        dst += (int(r[6]) + align - 1) // align * align
    with open(pack / "index.txt", "w", encoding="utf-8", newline="") as fo:
        fo.write("# strata pack index v3 -- repaired by tools/repair_heretic_pack.py\n")
        fo.write("# align %d pool %d tensors %d\n" % (align, dst, len(new_rows)))
        for r in new_rows:
            fo.write(" ".join(r) + "\n")
    print("index.txt rewritten: pool %d bytes (%.2f GiB), %d tensors" % (dst, dst / 2**30, len(new_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())