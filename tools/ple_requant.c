// tools/ple_requant.c - make the n-gram table the form Strata's PleTable accepts.
//
// WHY THIS EXISTS.  `src/kernels/ngram.cpp` accepts exactly one table: tensor
// `per_layer_token_embd.weight`, shape [160, n_rows], type **IQ4_NL**, 90 bytes per row
// (5 IQ4_NL blocks x 18 B).  Any other type is refused with "..., not IQ4_NL".  Community quants
// pin this table to Q8_0 (170 B/row), so their trunks load but their table does not: this tool
// reads the table out of any quantized GGUF and writes it back as IQ4_NL.
//
// IT USES GGML'S OWN QUANTIZER (`quantize_iq4_nl`, the ntry=7 path `llama-quantize` uses) from the
// pinned llama.cpp commit, so the block bytes are the reference producer's.
//
//   ple_requant --src <shard holding the table> --out ple-iq4nl.gguf [--verify ROWS] [--threads N]
//
// The output is a one-tensor GGUF whose data section holds exactly n_rows * 90 bytes, which is the
// invariant PleTable::open asserts for a table alone in its file.
#include "ggml.h"
#include "ggml-quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    char     name[128];
    uint64_t ne[4];
    uint32_t ndims;
    uint32_t type;
    uint64_t offset;
} TensorInfo;

static uint32_t rd_u32(FILE* f) { uint32_t v; if (fread(&v, 4, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); } return v; }
static uint64_t rd_u64(FILE* f) { uint64_t v; if (fread(&v, 8, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); } return v; }

static void rd_str(FILE* f, char* out, size_t cap) {
    const uint64_t n = rd_u64(f);
    const size_t take = n < cap - 1 ? (size_t) n : cap - 1;
    if (take && fread(out, take, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
    out[take] = 0;
    for (uint64_t i = take; i < n; ++i) { char c; if (fread(&c, 1, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); } }
}

static void skip_value(FILE* f, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: { char c; if (fread(&c,1,1,f)!=1){exit(1);} break; }
        case 2: case 3: { uint16_t v; if (fread(&v,2,1,f)!=1){exit(1);} break; }
        case 4: case 5: case 6: { uint32_t v; if (fread(&v,4,1,f)!=1){exit(1);} break; }
        case 10: case 11: case 12: { uint64_t v; if (fread(&v,8,1,f)!=1){exit(1);} break; }
        case 8: {
            const uint64_t n = rd_u64(f);
            char tmp[4096];
            uint64_t left = n;
            while (left) { const size_t take = left < sizeof tmp ? (size_t) left : sizeof tmp;
                           if (fread(tmp, 1, take, f) != take) { exit(1); } left -= take; }
            break;
        }
        case 9: {
            const uint32_t et = rd_u32(f);
            const uint64_t n = rd_u64(f);
            if (et == 8) { for (uint64_t i = 0; i < n; ++i) skip_value(f, 8); }
            else {
                size_t esz = 0;
                switch (et) { case 0: case 1: case 7: esz = 1; break; case 2: case 3: esz = 2; break;
                              case 4: case 5: case 6: esz = 4; break; case 10: case 11: case 12: esz = 8; break;
                              default: fprintf(stderr, "bad array elem type %u\n", et); exit(1); }
                const uint64_t bytes = esz * n;
                char tmp[4096];
                uint64_t left = bytes;
                while (left) { const size_t take = left < sizeof tmp ? (size_t) left : sizeof tmp;
                               if (fread(tmp, 1, take, f) != take) { exit(1); } left -= take; }
            }
            break;
        }
        default: fprintf(stderr, "unknown metadata type %u\n", type); exit(1);
    }
}

static size_t type_size(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32:    return 4;
        case GGML_TYPE_F16:    return 2;
        case GGML_TYPE_BF16:   return 2;
        case GGML_TYPE_Q4_0:   return sizeof(block_q4_0);
        case GGML_TYPE_Q8_0:   return sizeof(block_q8_0);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        default: return 0;
    }
}

static size_t type_block(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_IQ4_NL: return 32;
        default: return 0;
    }
}

static void dequant(uint32_t t, const void* src, float* dst, int64_t n) {
    switch (t) {
        case GGML_TYPE_F32:  memcpy(dst, src, (size_t) n * 4); break;
        case GGML_TYPE_F16:  { const ggml_fp16_t* h = (const ggml_fp16_t*) src; for (int64_t i=0;i<n;++i) dst[i] = ggml_fp16_to_fp32(h[i]); } break;
        case GGML_TYPE_BF16: { const uint16_t* h = (const uint16_t*) src; for (int64_t i=0;i<n;++i) { uint32_t w = (uint32_t) h[i] << 16; memcpy(&dst[i], &w, 4); } } break;
        case GGML_TYPE_Q4_0:   dequantize_row_q4_0((const block_q4_0*) src, dst, n); break;
        case GGML_TYPE_Q8_0:   dequantize_row_q8_0((const block_q8_0*) src, dst, n); break;
        case GGML_TYPE_IQ4_NL: dequantize_row_iq4_nl((const block_iq4_nl*) src, dst, n); break;
        default: fprintf(stderr, "unsupported source type %u\n", t); exit(1);
    }
}

// ---- GGUF v3 writer ------------------------------------------------------------------------------------
static void wr_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wr_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void wr_str(FILE* f, const char* s) { const uint64_t n = strlen(s); fwrite(&n, 8, 1, f); fwrite(s, 1, (size_t) n, f); }
static void wr_kv_str(FILE* f, const char* k, const char* v) { wr_str(f, k); wr_u32(f, 8); wr_str(f, v); }
static void wr_kv_u32(FILE* f, const char* k, uint32_t v) { wr_str(f, k); wr_u32(f, 4); wr_u32(f, v); }
static void wr_kv_arr_u64(FILE* f, const char* k, const uint64_t* v, uint64_t n) {
    wr_str(f, k); wr_u32(f, 9); wr_u32(f, 10); wr_u64(f, n);
    for (uint64_t i = 0; i < n; ++i) fwrite(&v[i], 8, 1, f);
}

static const uint64_t PLE_VOCAB[16] = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
static const uint64_t PLE_OFFSET[16] = {
    0,        20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

int main(int argc, char** argv) {
    const char* src_path = NULL;
    const char* out_path = NULL;
    int64_t verify_rows = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--src") && i + 1 < argc) src_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--verify") && i + 1 < argc) verify_rows = atoll(argv[++i]);
        else { fprintf(stderr, "usage: %s --src in.gguf --out out.gguf [--verify ROWS]\n", argv[0]); return 2; }
    }
    if (!src_path || !out_path) { fprintf(stderr, "--src and --out are required\n"); return 2; }

    FILE* f = fopen(src_path, "rb");
    if (!f) { perror(src_path); return 1; }
    if (rd_u32(f) != 0x46554747) { fprintf(stderr, "%s: not a GGUF\n", src_path); return 1; }
    const uint32_t version = rd_u32(f);
    const uint64_t n_tensors = rd_u64(f);
    const uint64_t n_kv = rd_u64(f);
    if (version != 3) { fprintf(stderr, "GGUF v%u; this tool handles v3\n", version); return 1; }
    printf("source: %s  v%u  %llu tensors  %llu kv\n", src_path, version,
           (unsigned long long) n_tensors, (unsigned long long) n_kv);

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        char key[256]; rd_str(f, key, sizeof key);
        const uint32_t t = rd_u32(f);
        if (!strcmp(key, "general.alignment") && t == 4) { alignment = rd_u32(f); continue; }
        skip_value(f, t);
    }

    TensorInfo ple; int found = 0;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo ti; rd_str(f, ti.name, sizeof ti.name);
        ti.ndims = rd_u32(f);
        if (ti.ndims > 4) { fprintf(stderr, "bad ndims\n"); return 1; }
        for (uint32_t d = 0; d < ti.ndims; ++d) ti.ne[d] = rd_u64(f);
        ti.type = rd_u32(f);
        ti.offset = rd_u64(f);
        if (!strcmp(ti.name, "per_layer_token_embd.weight")) { ple = ti; found = 1; }
    }
    if (!found) { fprintf(stderr, "per_layer_token_embd.weight is not in %s\n", src_path); return 1; }
    const long pos = ftell(f);
    const uint64_t src_data = ((uint64_t) pos + alignment - 1) / alignment * alignment;
    if (ple.ndims != 2 || ple.ne[0] != 160) {
        fprintf(stderr, "unexpected shape: ne = [%llu, %llu]\n", (unsigned long long) ple.ne[0],
                (unsigned long long) ple.ne[1]);
        return 1;
    }
    const uint64_t n_rows = ple.ne[1];
    const size_t tsz = type_size(ple.type), tblk = type_block(ple.type);
    if (!tsz || !tblk) { fprintf(stderr, "unsupported source tensor type %u\n", ple.type); return 1; }
    const size_t row_in = (size_t) (160 / tblk) * tsz;
    const size_t row_out = (size_t) (160 / 32) * sizeof(block_iq4_nl);
    printf("table: [160, %llu] ggml type %u, %zu B in / %zu B out per row\n",
           (unsigned long long) n_rows, ple.type, row_in, row_out);
    if (row_out != 90) { fprintf(stderr, "internal: row_out = %zu, expected 90\n", row_out); return 1; }
    {
        uint64_t total = 0;
        for (int h = 0; h < 16; ++h) total += PLE_VOCAB[h];
        if (total != n_rows) fprintf(stderr, "WARNING: the 16 head sizes sum to %llu, the table has %llu rows\n",
                                     (unsigned long long) total, (unsigned long long) n_rows);
    }

    FILE* o = fopen(out_path, "wb");
    if (!o) { perror(out_path); return 1; }
    wr_u32(o, 0x46554747); wr_u32(o, 3); wr_u64(o, 1); wr_u64(o, 11);
    wr_kv_str(o, "general.architecture", "qwen4exp");
    wr_kv_str(o, "general.name", "Qwen3.8-Flash-Next n-gram table (IQ4_NL, Strata PleTable form)");
    wr_kv_u32(o, "general.alignment", 4096);
    wr_kv_u32(o, "qwen4exp.embedding_length_per_layer_input", 160);
    wr_kv_u32(o, "qwen4exp.ple.ngram_size", 3);
    wr_kv_u32(o, "qwen4exp.ple.heads_per_ngram", 8);
    wr_kv_u32(o, "qwen4exp.ple.conv_kernel", 4);
    wr_kv_u32(o, "qwen4exp.ple.eos_token_id", 248044);
    wr_kv_arr_u64(o, "qwen4exp.ple.layers", (const uint64_t[]){1}, 1);
    wr_kv_arr_u64(o, "qwen4exp.ple.head_vocab_sizes", PLE_VOCAB, 16);
    wr_kv_arr_u64(o, "qwen4exp.ple.head_offsets", PLE_OFFSET, 16);
    wr_str(o, "per_layer_token_embd.weight"); wr_u32(o, 2);
    { const uint64_t ne[2] = {160, n_rows}; fwrite(ne, 8, 2, o); }
    wr_u32(o, GGML_TYPE_IQ4_NL);
    wr_u64(o, 0);
    fflush(o);
    const long opos = ftell(o);
    const uint64_t out_data = ((uint64_t) opos + 4095) / 4096 * 4096;
    for (long i = opos; i < (long) out_data; ++i) fputc(0, o);
    printf("output data section starts at %llu (header %ld, alignment 4096)\n",
           (unsigned long long) out_data, opos);

    // ---- the pass itself ------------------------------------------------------------------------------
    const int64_t CHUNK = 1 << 16;                       // rows per IO chunk (10.6 MB in, 5.6 MB out)
    float* buf = (float*) malloc((size_t) CHUNK * 160 * sizeof(float));
    block_iq4_nl* q = (block_iq4_nl*) malloc((size_t) CHUNK * 5 * sizeof(block_iq4_nl));
    unsigned char* raw_in = (unsigned char*) malloc((size_t) CHUNK * row_in);
    if (!buf || !q || !raw_in) { fprintf(stderr, "allocation failed\n"); return 1; }

    uint64_t written = 0;
    const uint64_t n_chunks = (n_rows + (uint64_t) CHUNK - 1) / (uint64_t) CHUNK;
    for (uint64_t r0 = 0; r0 < n_rows; r0 += (uint64_t) CHUNK) {
        const uint64_t rows = (n_rows - r0 < (uint64_t) CHUNK) ? (n_rows - r0) : (uint64_t) CHUNK;
        if (fseeko(f, (off_t) (src_data + ple.offset + r0 * row_in), SEEK_SET) != 0) { perror("seek"); return 1; }
        if (fread(raw_in, row_in, (size_t) rows, f) != (size_t) rows) {
            fprintf(stderr, "short read at row %llu\n", (unsigned long long) r0);
            return 1;
        }
        for (uint64_t i = 0; i < rows; ++i) {
            dequant(ple.type, raw_in + i * row_in, buf + i * 160, 160);
            quantize_iq4_nl(buf + i * 160, (void*) (q + i * 5), 1, 160, NULL);
        }
        if (fwrite(q, (size_t) row_out, (size_t) rows, o) != (size_t) rows) { fprintf(stderr, "short write\n"); return 1; }
        written += rows;
        if ((r0 / CHUNK) % 128 == 0 || r0 + rows == n_rows) {
            printf("  %6.2f%%  %llu/%llu rows\n", 100.0 * (double) written / (double) n_rows,
                   (unsigned long long) written, (unsigned long long) n_rows);
            fflush(stdout);
        }
        (void) n_chunks;
    }
    if (fclose(o) != 0) { perror("close"); return 1; }
    const uint64_t expect = out_data + n_rows * row_out;
    { FILE* chk = fopen(out_path, "rb"); fseeko(chk, 0, SEEK_END); const uint64_t actual = (uint64_t) ftello(chk); fclose(chk);
      printf("wrote %s: %llu bytes (expected %llu), %.2f GiB\n", out_path, (unsigned long long) actual,
             (unsigned long long) expect, (double) actual / 1073741824.0);
      if (actual != expect) { fprintf(stderr, "SIZE MISMATCH - the engine would refuse this table\n"); return 1; } }

    // ---- verification: dequantize what we wrote and compare with the source's own values ---------------
    if (verify_rows > 0) {
        const int64_t N = verify_rows < (int64_t) n_rows ? verify_rows : (int64_t) n_rows;
        float* a = (float*) malloc((size_t) N * 160 * sizeof(float));
        float* b = (float*) malloc((size_t) N * 160 * sizeof(float));
        unsigned char* rq = (unsigned char*) malloc((size_t) N * row_out);
        unsigned char* rs = (unsigned char*) malloc((size_t) N * row_in);
        FILE* v = fopen(out_path, "rb");
        if (!a || !b || !rq || !rs || !v) { fprintf(stderr, "verify: allocation/open failed\n"); return 1; }
        {   // find the output tensor's data offset (header is ours: one tensor at offset 0)
            uint32_t m2 = rd_u32(v); (void) m2; (void) rd_u32(v);
            const uint64_t nt = rd_u64(v), nk = rd_u64(v);
            uint64_t al = 32;
            for (uint64_t i = 0; i < nk; ++i) {
                char key[256]; rd_str(v, key, sizeof key);
                const uint32_t t = rd_u32(v);
                if (!strcmp(key, "general.alignment") && t == 4) { al = rd_u32(v); continue; }
                skip_value(v, t);
            }
            long p = 0;
            for (uint64_t i = 0; i < nt; ++i) {
                char nm[128]; rd_str(v, nm, sizeof nm);
                const uint32_t nd = rd_u32(v);
                for (uint32_t d = 0; d < nd; ++d) (void) rd_u64(v);
                (void) rd_u32(v); (void) rd_u64(v);
                p = ftell(v);
            }
            const uint64_t od = ((uint64_t) p + al - 1) / al * al;
            fseeko(v, (off_t) od, SEEK_SET);
        }
        fseeko(f, (off_t) (src_data + ple.offset), SEEK_SET);
        if (fread(rs, row_in, (size_t) N, f) != (size_t) N) { fprintf(stderr, "verify: short src read\n"); return 1; }
        if (fread(rq, row_out, (size_t) N, v) != (size_t) N) { fprintf(stderr, "verify: short out read\n"); return 1; }
        double max_abs = 0, sum_abs = 0, sum_src = 0, sum_d2 = 0, sum_s2 = 0;
        for (int64_t i = 0; i < N; ++i) {
            dequant(ple.type, rs + (size_t) i * row_in, a + (size_t) i * 160, 160);
            dequantize_row_iq4_nl((const block_iq4_nl*) (rq + (size_t) i * row_out), b + (size_t) i * 160, 160);
            for (int j = 0; j < 160; ++j) {
                const double d = (double) b[(size_t) i * 160 + j] - (double) a[(size_t) i * 160 + j];
                if (fabs(d) > max_abs) max_abs = fabs(d);
                sum_abs += fabs(d); sum_src += fabs((double) a[(size_t) i * 160 + j]);
                sum_d2 += d * d; sum_s2 += (double) a[(size_t) i * 160 + j] * (double) a[(size_t) i * 160 + j];
            }
        }
        printf("verify on %lld rows: max|err| %.6g   mean|err| %.6g (mean|src| %.6g)   rel L2 %.4f%%\n",
               (long long) N, max_abs, sum_abs / (N * 160), sum_src / (N * 160),
               100.0 * sqrt(sum_d2 / (sum_s2 > 0 ? sum_s2 : 1)));
    }
    fclose(f);
    free(buf); free(q); free(raw_in);
    return 0;
}
