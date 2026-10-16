#!/usr/bin/env python3
"""tools/make_ple_iq4nl.py - the n-gram table in the one form Strata's PleTable accepts.

`src/kernels/ngram.cpp` accepts exactly: tensor `per_layer_token_embd.weight`, shape [160, n_rows],
type **IQ4_NL**, 90 bytes per row (5 blocks of 18 B).  Community quants pin this table to Q8_0
(170 B/row), so the trunk loads and the table is refused.  This tool reads the table out of any
quantized GGUF and writes a one-tensor GGUF holding it as IQ4_NL.

The quantizer is a vectorized transcription of ggml's `quantize_iq4_nl` (ggml-quants.c, the ntry=7
path `llama-quantize` uses): weights x^2, initial d = max/values[0], then 15 candidate scales
`id = (itry + values[0])/max` keeping the best by `sumqx^2 > best*sumq2`, and the codes taken at
`id = 1/d_final`.  tools/ple_requant.c calls the real ggml function for byte-level cross-checks.

    python3 make_ple_iq4nl.py --src <shard holding the table> --out ple-iq4nl.gguf [--verify N]
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

# ggml's IQ4_NL table (ggml-quants.c: kvalues_iq4nl); ngram.cpp's kIq4Nl holds the same 16 numbers
KVALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                   dtype=np.int8)
QK = 32
NBLK = 160 // QK            # 5 blocks per row
ROW_OUT = NBLK * 18         # 90 bytes
GROUP_MAX_EPS = 1e-9

PLE_VOCAB = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
             20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
PLE_OFFSET = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
              160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

SRC_KIND = {  # type name -> (bytes per block, values per block, structured dtype of one block)
    "Q8_0": (34, 32, np.dtype([("d", "<f2"), ("qs", "i1", 32)])),
    "F16": (2, 1, np.dtype("<f2")),
    "BF16": (2, 1, np.dtype("<u2")),
    "F32": (4, 1, np.dtype("<f4")),
}


def best_index_int8(x: np.ndarray) -> np.ndarray:
    """ggml's `best_index_int8(16, kvalues_iq4nl, x)`, vectorized (ties go to the upper index)."""
    hi = np.clip(np.searchsorted(KVALUES, x, side="left").astype(np.int64), 0, 15)
    lo = np.clip(hi - 1, 0, 15)
    return np.where(x - KVALUES[lo] < KVALUES[hi] - x, lo, hi).astype(np.uint8)


def quantize_iq4_nl(x: np.ndarray) -> np.ndarray:
    """x: (M, 5, 32) float32 -> (M, 5, 18) uint8, ggml's quantize_iq4_nl semantics."""
    m = x.shape[0]
    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = x ** 2                                                              # (M, 5, 32)
    absx = np.abs(x)
    amax = np.max(absx, axis=2)                                                  # (M, 5)
    flat = np.argmax(absx, axis=2)
    mx = np.take_along_axis(x, flat[:, :, None], axis=2)[:, :, 0]                 # the signed |max| value
    live = amax >= GROUP_MAX_EPS

    def pack(d: np.ndarray, L: np.ndarray) -> np.ndarray:
        out = np.zeros((m, NBLK, 18), dtype=np.uint8)
        out[:, :, 0:2] = np.frombuffer(np.ascontiguousarray(d, dtype="<f2").tobytes(),
                                       dtype=np.uint8).reshape(m, NBLK, 2)
        out[:, :, 2:18] = (L[:, :, 0:16] | (L[:, :, 16:32] << 4)).astype(np.uint8)
        return out

    if not np.any(live):
        return pack(np.zeros((m, NBLK), dtype=np.float32), np.zeros((m, NBLK, 32), dtype=np.uint8))

    safe_max = np.where(live, mx, 1.0).astype(np.float32)
    d = np.where(live, safe_max / np.float32(KVALUES[0]), np.float32(0.0)).astype(np.float32)
    rid = np.where(live & (d != 0), np.float32(1.0) / np.where(d != 0, d, np.float32(1.0)),
                   np.float32(0.0)).astype(np.float32)

    def accumulate(idv: np.ndarray, L: np.ndarray):
        q = KVALUES[L].astype(np.float32)
        return (np.sum(weight * q * x, axis=2, dtype=np.float64),
                np.sum(weight * q * q, axis=2, dtype=np.float64))

    L = best_index_int8(rid[:, :, None] * x)
    sumqx, sumq2 = accumulate(rid, L)
    d = np.where(live & (sumq2 > 0), (sumqx / np.where(sumq2 > 0, sumq2, 1.0)).astype(np.float32), d)
    d = np.ascontiguousarray(d, dtype=np.float32)
    best = d.astype(np.float64) * sumqx

    for itry in range(-7, 8):
        idv = ((itry + int(KVALUES[0])) / safe_max).astype(np.float32)
        Lc = best_index_int8(idv[:, :, None] * x)
        sqx, sq2 = accumulate(idv, Lc)
        upd = live & (sq2 > 0) & (sqx * sqx > best * sq2)
        d = np.where(upd, (sqx / np.where(sq2 > 0, sq2, 1.0)).astype(np.float32), d)
        best = np.where(upd, sqx * sqx / np.where(sq2 > 0, sq2, 1.0), best)
    d = np.ascontiguousarray(d, dtype=np.float32)

    idf = np.where(live & (d != 0), np.float32(1.0) / np.where(d != 0, d, np.float32(1.0)),
                   np.float32(0.0)).astype(np.float32)
    L = best_index_int8(idf[:, :, None] * x)
    return pack(np.where(live, d, np.float32(0.0)), L)


def wr_u32(f, v): f.write(struct.pack("<I", v))
def wr_u64(f, v): f.write(struct.pack("<Q", v))
def wr_str(f, s): f.write(struct.pack("<Q", len(s))); f.write(s.encode())
def wr_kv_str(f, k, v): wr_str(f, k); wr_u32(f, 8); wr_str(f, v)
def wr_kv_u32(f, k, v): wr_str(f, k); wr_u32(f, 4); wr_u32(f, v)
def wr_kv_arr_u64(f, k, vals):
    wr_str(f, k); wr_u32(f, 9); wr_u32(f, 10); wr_u64(f, len(vals))
    for v in vals: wr_u64(f, v)


def dequant_source(kind_name, raw, rows, blocks_per_row, dtype):
    """(rows, 160) float32 from the raw bytes of one chunk of rows."""
    if kind_name == "Q8_0":
        b = raw.view(dtype).reshape(rows, blocks_per_row)
        return (b["d"].astype(np.float32)[:, :, None] * b["qs"].astype(np.float32)).reshape(rows, 5, 32)
    if kind_name == "BF16":
        v = raw.view(np.uint16).reshape(rows, 160)
        return (v.astype(np.uint32) << 16).view(np.int32).view(np.float32).reshape(rows, 5, 32)
    if kind_name == "F16":
        return raw.view(np.float16).astype(np.float32).reshape(rows, 5, 32)
    return raw.view(np.float32).reshape(rows, 5, 32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1 << 20, help="rows per chunk")
    ap.add_argument("--verify", type=int, default=4096, help="rows to re-check after writing")
    a = ap.parse_args()

    src = pathlib.Path(a.src)
    g = G.GGUFFile(src)
    t = next((x for x in g.tensors if x.name == "per_layer_token_embd.weight"), None)
    if t is None:
        print(f"{src}: per_layer_token_embd.weight is not in this file")
        return 1
    if t.shape[0] != 160:
        print(f"unexpected shape {t.shape}; ne0 must be 160")
        return 1
    n_rows = int(t.shape[1])
    if t.type_name == "IQ4_NL":
        print(f"{src}: the table is ALREADY IQ4_NL ({n_rows} rows) - nothing to do")
        return 0
    kind = SRC_KIND.get(t.type_name)
    if kind is None:
        print(f"{src}: cannot read a {t.type_name} table (supported: {', '.join(SRC_KIND)})")
        return 1
    blk_bytes, blk_vals, dtype = kind
    per_row_in = 160 // blk_vals * blk_bytes
    blocks_per_row = 160 // blk_vals
    print(f"source: {src.name}  table [160, {n_rows}] {t.type_name} "
          f"({per_row_in} B/row in, {ROW_OUT} B/row out)")
    if sum(PLE_VOCAB) != n_rows:
        print(f"WARNING: the 16 head sizes sum to {sum(PLE_VOCAB)}, the table has {n_rows} rows")

    out = pathlib.Path(a.out)
    fo = out.open("wb")
    wr_u32(fo, 0x46554747); wr_u32(fo, 3); wr_u64(fo, 1); wr_u64(fo, 11)
    wr_kv_str(fo, "general.architecture", "qwen4exp")
    wr_kv_str(fo, "general.name", "Qwen3.8-Flash-Next n-gram table (IQ4_NL, Strata PleTable form)")
    wr_kv_u32(fo, "general.alignment", 4096)
    wr_kv_u32(fo, "qwen4exp.embedding_length_per_layer_input", 160)
    wr_kv_u32(fo, "qwen4exp.ple.ngram_size", 3)
    wr_kv_u32(fo, "qwen4exp.ple.heads_per_ngram", 8)
    wr_kv_u32(fo, "qwen4exp.ple.conv_kernel", 4)
    wr_kv_u32(fo, "qwen4exp.ple.eos_token_id", 248044)
    wr_kv_arr_u64(fo, "qwen4exp.ple.layers", [1])
    wr_kv_arr_u64(fo, "qwen4exp.ple.head_vocab_sizes", PLE_VOCAB)
    wr_kv_arr_u64(fo, "qwen4exp.ple.head_offsets", PLE_OFFSET)
    wr_str(fo, "per_layer_token_embd.weight"); wr_u32(fo, 2); wr_u64(fo, 160); wr_u64(fo, n_rows)
    wr_u32(fo, 20)                                  # ggml type 20 = IQ4_NL
    wr_u64(fo, 0)
    hdr_end = fo.tell()
    fo.write(b"\0" * ((-hdr_end) % 4096))
    data_start = fo.tell()
    print(f"header {hdr_end} B, data starts at {data_start}")

    mm = np.memmap(src, dtype=np.uint8, mode="r")
    base = g.data_start + t.offset
    done = 0
    t0 = time.time()
    for r0 in range(0, n_rows, a.chunk):
        rows = min(a.chunk, n_rows - r0)
        raw = np.asarray(mm[base + r0 * per_row_in: base + (r0 + rows) * per_row_in])
        x = dequant_source(t.type_name, raw, rows, blocks_per_row, dtype)
        fo.write(quantize_iq4_nl(x).tobytes())
        done += rows
        print(f"  {100.0 * done / n_rows:6.2f}%  {done}/{n_rows} rows  "
              f"{done * ROW_OUT / 2**30:.2f} GiB written  {(time.time() - t0) / 60:.1f} min", flush=True)
    fo.close()
    expect = data_start + n_rows * ROW_OUT
    print(f"wrote {out}: {out.stat().st_size} bytes (expected {expect})")
    if out.stat().st_size != expect:
        print("SIZE MISMATCH - the engine would refuse this table")
        return 1

    if a.verify > 0:
        N = min(a.verify, n_rows)
        g2 = G.GGUFFile(out)
        t2 = next(x for x in g2.tensors if x.name == "per_layer_token_embd.weight")
        g2_size = out.stat().st_size
        print(f"verify: {out.name} reports {t2.type_name} {list(t2.shape)}, "
              f"{g2_size - g2.data_start} B of data for {n_rows * ROW_OUT} B of rows")
        a_ref = dequant_source(t.type_name, np.asarray(mm[base: base + N * per_row_in]),
                               N, blocks_per_row, dtype)
        ob = np.asarray(np.memmap(out, dtype=np.uint8, mode="r")[g2.data_start: g2.data_start + N * ROW_OUT])
        ob = ob.view(np.dtype([("d", "<f2"), ("qs", "u1", 16)])).reshape(N, 5)
        lo = KVALUES[ob["qs"] & 0x0F].astype(np.float32)
        hi = KVALUES[ob["qs"] >> 4].astype(np.float32)
        b_val = ob["d"].astype(np.float32)[:, :, None] * np.concatenate([lo, hi], axis=2)
        diff = b_val - a_ref
        rel = 100.0 * np.sqrt(np.sum(diff ** 2) / np.sum(a_ref ** 2))
        print(f"verify {N} rows: max|err| {np.abs(diff).max():.4g}  mean|err| {np.abs(diff).mean():.4g} "
              f"(mean|src| {np.abs(a_ref).mean():.4g})  rel L2 {rel:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

