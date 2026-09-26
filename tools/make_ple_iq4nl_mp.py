#!/usr/bin/env python3
"""make_ple_iq4nl_mp.py - parallel variant of tools/make_ple_iq4nl.py (identical output bytes).

The stock tool is single-threaded numpy and takes ~5 h on this machine for the 320,001,536-row
Q8_0 n-gram table; this fork keeps the exact per-row math (same functions imported from the
original module) and spreads chunks over N worker processes, each writing its slice of the
output file at the correct offset.  The result is renamed into place only when complete, so a
concurrent `heretic_finish.sh` never sees a half-written table.

    python tools/make_ple_iq4nl_mp.py --src <shard3.gguf> --out ple/ple-iq4nl.gguf \
        --workers 12 --chunk 262144 [--verify 1024]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pathlib
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G                      # noqa: E402
import make_ple_iq4nl as base                # noqa: E402

W = {}                                       # worker globals


def _init(src, base_off, per_row_in, blocks_per_row, dtype, ttype, out_path, data_start, row_out):
    W["src"] = src
    W["mm"] = np.memmap(src, dtype=np.uint8, mode="r")
    W["base"] = base_off
    W["per_row_in"] = per_row_in
    W["blocks_per_row"] = blocks_per_row
    W["dtype"] = dtype
    W["ttype"] = ttype
    W["out"] = out_path
    W["data_start"] = data_start
    W["row_out"] = row_out


def _work(args):
    r0, rows = args
    mm, per_row_in = W["mm"], W["per_row_in"]
    raw = np.asarray(mm[W["base"] + r0 * per_row_in: W["base"] + (r0 + rows) * per_row_in])
    x = base.dequant_source(W["ttype"], raw, rows, W["blocks_per_row"], W["dtype"])
    buf = base.quantize_iq4_nl(x).tobytes()
    assert len(buf) == rows * W["row_out"]
    with open(W["out"], "r+b") as fo:
        fo.seek(W["data_start"] + r0 * W["row_out"])
        fo.write(buf)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=262144, help="rows per chunk (memory: ~4x chunk x 320 floats)")
    ap.add_argument("--verify", type=int, default=1024)
    a = ap.parse_args()

    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    tmp = out.with_name(out.name + ".mp.tmp")
    g = G.GGUFFile(src)
    t = next((x for x in g.tensors if x.name == "per_layer_token_embd.weight"), None)
    if t is None or t.shape[0] != 160:
        print(f"{src}: per_layer_token_embd.weight missing or wrong shape {None if t is None else t.shape}")
        return 1
    n_rows = int(t.shape[1])
    if t.type_name == "IQ4_NL":
        print(f"{src}: already IQ4_NL, nothing to do")
        return 0
    kind = base.SRC_KIND.get(t.type_name)
    if kind is None:
        print(f"{src}: cannot read a {t.type_name} table")
        return 1
    blk_bytes, blk_vals, dtype = kind
    per_row_in = 160 // blk_vals * blk_bytes
    blocks_per_row = 160 // blk_vals
    print(f"source: {src.name}  table [160, {n_rows}] {t.type_name} "
          f"({per_row_in} B/row in, {base.ROW_OUT} B/row out), workers={a.workers} chunk={a.chunk}")

    # ---- header (identical to the stock tool)
    fo = tmp.open("wb")
    base.wr_u32(fo, 0x46554747); base.wr_u32(fo, 3); base.wr_u64(fo, 1); base.wr_u64(fo, 11)
    base.wr_kv_str(fo, "general.architecture", "qwen4exp")
    base.wr_kv_str(fo, "general.name", "Qwen3.8-Flash-Next n-gram table (IQ4_NL, Strata PleTable form)")
    base.wr_kv_u32(fo, "general.alignment", 4096)
    base.wr_kv_u32(fo, "qwen4exp.embedding_length_per_layer_input", 160)
    base.wr_kv_u32(fo, "qwen4exp.ple.ngram_size", 3)
    base.wr_kv_u32(fo, "qwen4exp.ple.heads_per_ngram", 8)
    base.wr_kv_u32(fo, "qwen4exp.ple.conv_kernel", 4)
    base.wr_kv_u32(fo, "qwen4exp.ple.eos_token_id", 248044)
    base.wr_kv_arr_u64(fo, "qwen4exp.ple.layers", [1])
    base.wr_kv_arr_u64(fo, "qwen4exp.ple.head_vocab_sizes", base.PLE_VOCAB)
    base.wr_kv_arr_u64(fo, "qwen4exp.ple.head_offsets", base.PLE_OFFSET)
    base.wr_str(fo, "per_layer_token_embd.weight"); base.wr_u32(fo, 2); base.wr_u64(fo, 160); base.wr_u64(fo, n_rows)
    base.wr_u32(fo, 20)
    base.wr_u64(fo, 0)
    hdr_end = fo.tell()
    fo.write(b"\0" * ((-hdr_end) % 4096))
    data_start = fo.tell()
    fo.truncate(data_start + n_rows * base.ROW_OUT)
    fo.close()
    print(f"header {hdr_end} B, data starts at {data_start}")

    chunks = [(r0, min(a.chunk, n_rows - r0)) for r0 in range(0, n_rows, a.chunk)]
    t0 = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(str(src), g.data_start + t.offset, per_row_in, blocks_per_row,
                            dtype, t.type_name, str(tmp), data_start, base.ROW_OUT)) as pool:
        for rows in pool.imap_unordered(_work, chunks, chunksize=1):
            done += rows
            if done % (a.chunk * 20) < a.chunk:
                el = (time.time() - t0) / 60
                print(f"  {100.0 * done / n_rows:6.2f}%  {done}/{n_rows} rows  "
                      f"{done * base.ROW_OUT / 2**30:.2f} GiB  {el:.1f} min  "
                      f"(eta {el / max(done, 1) * (n_rows - done):.0f} min)", flush=True)
    expect = data_start + n_rows * base.ROW_OUT
    assert tmp.stat().st_size == expect, (tmp.stat().st_size, expect)
    os.replace(tmp, out)
    print(f"wrote {out}: {expect} bytes, {n_rows} rows, in {(time.time() - t0) / 60:.1f} min")

    if a.verify > 0:
        N = min(a.verify, n_rows)
        g2 = G.GGUFFile(out)
        t2 = next(x for x in g2.tensors if x.name == "per_layer_token_embd.weight")
        print(f"verify: {out.name} reports {t2.type_name} {list(t2.shape)}")
        mm = np.memmap(src, dtype=np.uint8, mode="r")
        base_off = g.data_start + t.offset
        a_ref = base.dequant_source(t.type_name, np.asarray(mm[base_off: base_off + N * per_row_in]),
                                    N, blocks_per_row, dtype)
        ob = np.asarray(np.memmap(out, dtype=np.uint8, mode="r")[g2.data_start: g2.data_start + N * base.ROW_OUT])
        ob = ob.view(np.dtype([("d", "<f2"), ("qs", "u1", 16)])).reshape(N, 5)
        lo = base.KVALUES[ob["qs"] & 0x0F].astype(np.float32)
        hi = base.KVALUES[ob["qs"] >> 4].astype(np.float32)
        b_val = ob["d"].astype(np.float32)[:, :, None] * np.concatenate([lo, hi], axis=2)
        diff = b_val - a_ref
        rel = 100.0 * np.sqrt(np.sum(diff ** 2) / np.sum(a_ref ** 2))
        print(f"verify {N} rows: max|err| {np.abs(diff).max():.4g}  mean|err| {np.abs(diff).mean():.4g} "
              f"(mean|src| {np.abs(a_ref).mean():.4g})  rel L2 {rel:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())