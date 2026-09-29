// bench/micro/grouped_mm_bench.cu
//
// R16: the expert kernel is a GEMV, and that is what is left on the table.
//
// `s2_expert_grouped.cu`'s grid is `n_hits * 2FF` - one warp per (token-hit, weight row) - so for a chunk of C
// tokens every expert's weight row is walked once per token that routed to it.  Weight traffic per token is
// therefore CONSTANT, independent of chunk size and independent of how much VRAM you add:
//
//     weight bytes/token = 48 layers x 1.04 GB/layer / 64 tokens ~= 780 MB/token
//
// which at 504 GB/s of GDDR6 is a ~645 tok/s ceiling that no config knob in the campaign can raise.  The same
// arithmetic caps DECODE: the resident tier is consumed by the CPU pool at ~60 GB/s, so 1.044 GB per token is
// ~17 ms before a single miss is paid.
//
// THE FIX IS INDEXING, NOT HARDWARE.  Sort a window's hits by expert and give each warp ONE weight row and
// MANY tokens.  The row's codes are loaded once, unpacked once, and reused across every token routed to that
// expert, so weight traffic falls by the tokens-per-expert factor and what remains is the activation dot
// products - the part that has to be done anyway.
//
// THE SECOND, FREE WIN: `hx`.  The kernel computes
//     acc += d_w * d_x * ( sum_j c_j*x_j - sum_j x_j )
// with TWO dp4a per 32 elements - one for the codes, one for `sum_j x_j`.  The second term does not depend on
// the weight at all, so recomputing it for every row of every expert is pure waste.  Hoisting it into a
// per-token array halves the integer work.
//
// NUMERICS: the lane->chunk mapping (lane L owns chunks L, L+32, ...), the `hx` integer type, and the shuffle
// reduction are kept EXACTLY as the shipped kernel, so every (row, token) product is produced by the same
// sequence of operations and the two kernels must agree to the BIT.  That is what makes it safe to drop into a
// path whose parity harness caught a 2.2e+03 row-interleaving error on its first run.
//
// build: nvcc -O3 -arch=sm_89 -o /tmp/gmb grouped_mm_bench.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CK(x)                                                                                                \
    do {                                                                                                     \
        cudaError_t e_ = (x);                                                                                \
        if (e_ != cudaSuccess) {                                                                             \
            fprintf(stderr, "cuda: %s @%d\n", cudaGetErrorString(e_), __LINE__);                             \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

// THE BLOB'S GEOMETRY, restated as literals exactly as `s2_expert_grouped.cu` does, so this file cannot
// silently follow a header change and start measuring a different layout.
constexpr int H = 2560;              // expert hidden size
constexpr int FF = 640;              // expert intermediate size
constexpr int QK = 64;               // one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;        // 640 B of 2-bit codes per gate/up row
constexpr int ROW_D = FF / 4;        // 160 B per down row
constexpr int SC_GU = H / QK;        // 40 fp16 scales per gate/up row
constexpr int NCH = H / 32;          // 80 chunks of 32 elements per gate/up row
constexpr int ROWS_GU = 2 * FF;      // 1280 gate/up rows per expert
constexpr size_t O_D_CODES = (size_t) ROWS_GU * ROW_GU;
constexpr size_t O_GU_CODES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_GU_SCALES = O_GU_CODES + (size_t) ROWS_GU * ROW_GU;
constexpr size_t BLOB = 2181120;     // the real max_blob, so stride and hence the DRAM pattern match

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int TOPK = 8;
constexpr int XROW = NCH * 34;       // bytes of one token's Q8_0 activation

__device__ __forceinline__ float f16_at(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------- THE SHIPPED ARITHMETIC, verbatim
/// One row against one Q8_0 activation, exactly `row_dot_s2_q8`.  `hx` is recomputed per row here - that is
/// the waste the tiled kernel removes.
__device__ float row_dot_ref(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                             const uint8_t* __restrict__ x, int lane) {
    float acc = 0.0f;
    const int ones = 0x01010101;
    for (int c = lane; c < NCH; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;
        const uint8_t* xb = x + (size_t) c * 34;
        const float dx = f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);
        int s = 0, hx = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned b = cb[j];
            const int cw = (int) ((b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) |
                                  (((b >> 6) & 3u) << 24));
            int xw;
            memcpy(&xw, xq + 4 * j, 4);        // 34-byte blocks leave x unaligned; a cast faults
            s = __dp4a(cw, xw, s);
            hx = __dp4a(ones, xw, hx);
        }
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

__global__ void gu_ref(const uint8_t* __restrict__ blob, const int32_t* __restrict__ slot_of_hit,
                       const uint8_t* __restrict__ x, const int32_t* __restrict__ tok_of_hit, int n_hits,
                       float* __restrict__ gate_up) {
    const long long slot = (long long) blockIdx.x * WARPS + (threadIdx.x >> 5);
    const long long total = (long long) n_hits * ROWS_GU;
    if (slot >= total) return;
    const int h = (int) (slot / ROWS_GU);
    const int i = (int) (slot % ROWS_GU);
    const int lane = threadIdx.x & 31;
    const uint8_t* b = blob + (size_t) slot_of_hit[h] * BLOB;
    const float acc = row_dot_ref(b + O_GU_CODES + (size_t) i * ROW_GU,
                                  b + O_GU_SCALES + (size_t) i * SC_GU * 2,
                                  x + (size_t) tok_of_hit[h] * XROW, lane);
    const float s = warp_sum(acc);
    if (lane != 0) return;
    // gate-major: the shipped contract, and the one its parity test caught getting wrong (worst relative error
    // 2.2e+03 when row-slot i was written to output slot i).  Row-slot i is gate row i/2 if even, up row
    // (i-1)/2 if odd.
    const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
    gate_up[base + (size_t) (i >> 1)] = s;
}

// ---------------------------------------------------------------- THE GROUPED REPLACEMENT
/// `hx` per token per chunk - weight-independent, so computed once per token instead of once per row of every
/// expert the token routes to.
__global__ void hx_kernel(const uint8_t* __restrict__ x, int n_tok, int* __restrict__ hx) {
    const int t = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (t >= n_tok) return;
    const int lane = threadIdx.x & 31;
    const uint8_t* xb = x + (size_t) t * XROW;
    const int ones = 0x01010101;
    for (int c = lane; c < NCH; c += 32) {
        const int8_t* xq = (const int8_t*) (xb + (size_t) c * 34 + 2);
        int s = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int xw;
            memcpy(&xw, xq + 4 * j, 4);
            s = __dp4a(ones, xw, s);
        }
        hx[(size_t) t * NCH + c] = s;
    }
}

constexpr int TM = 8;              // tokens per inner tile
constexpr int ROWS_PER_BLOCK = ROWS_GU / WARPS;

/// One warp = ONE weight row of ONE expert; the loop is over every token routed to that expert.  The row's
/// codes are loaded and unpacked ONCE per chunk and reused across the tile, and `hx` is precomputed, so per
/// useful product the reference's 16 dp4a + 8 code unpacks become 8 dp4a and a shared unpack, and weight
/// traffic drops by the tokens-per-expert factor.
__global__ void gu_tiled(const uint8_t* __restrict__ blob, const uint8_t* __restrict__ x,
                         const int* __restrict__ hx, const int32_t* __restrict__ begin,
                         const int32_t* __restrict__ hit_tok, const int32_t* __restrict__ hit_id, int n_hits,
                         float* __restrict__ gate_up) {
    const int e = (int) (blockIdx.x / ROWS_PER_BLOCK);
    const int i = (int) (blockIdx.x % ROWS_PER_BLOCK) * WARPS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;

    const int b0 = begin[e], b1 = begin[e + 1];
    if (b0 >= b1) return;                       // nothing routed to this expert this window
    const uint8_t* eb = blob + (size_t) e * BLOB;
    const uint8_t* codes = eb + O_GU_CODES + (size_t) i * ROW_GU;
    const uint8_t* scales = eb + O_GU_SCALES + (size_t) i * SC_GU * 2;

    for (int base = b0; base < b1; base += TM) {
        const int m = min(TM, b1 - base);
        float acc[TM];
#pragma unroll
        for (int t = 0; t < TM; ++t) acc[t] = 0.0f;

        int32_t hid[TM];
        const uint8_t* xp[TM];
        const int* hp[TM];
        for (int t = 0; t < m; ++t) {
            hid[t] = hit_id[base + t];
            xp[t] = x + (size_t) hit_tok[base + t] * XROW;
            hp[t] = hx + (size_t) hit_tok[base + t] * NCH;
        }
        for (int c = lane; c < NCH; c += 32) {
            const uint8_t* cb = codes + (size_t) c * 8;
            int cw[8];                          // unpacked ONCE for the whole tile
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned b = cb[j];
                cw[j] = (int) ((b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) |
                               (((b >> 6) & 3u) << 24));
            }
            const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
            for (int t = 0; t < m; ++t) {
                const uint8_t* xb = xp[t] + (size_t) c * 34;
                const float dx = f16_at(xb);
                const int8_t* xq = (const int8_t*) (xb + 2);
                int s = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int xw;
                    memcpy(&xw, xq + 4 * j, 4);
                    s = __dp4a(cw[j], xw, s);
                }
                acc[t] += dw * dx * (float) (s - hp[t][c]);   // still integer subtraction: bit-exact
            }
        }
        for (int t = 0; t < m; ++t) {
            const float s = warp_sum(acc[t]);
            if (lane != 0) continue;
            const int h = hid[t];
            const size_t o = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
            gate_up[o + (size_t) (i >> 1)] = s;
        }
    }
}

// ------------------------------------------------- v2: amortize the ACTIVATION across ROWS
// The 0.93x above is the real finding: cutting weight traffic 128x changes nothing, because the weights are
// L2-resident.  What the kernel is actually short of is useful work per load - for every (row, token, chunk) it
// issues EIGHT unaligned 4-byte loads of the activation to feed EIGHT dp4a, a 1:1 ratio that leaves the SM
// issuing loads.  So the operand to amortize is the ACTIVATION: hold one token-chunk in registers and spend it
// on R different weight rows.  Ratio goes to 8:1 and the codes stay resident in registers across the tile.
constexpr int R = 8;                                    // rows per warp
constexpr int TM2 = 4;                                  // tokens per tile
constexpr int RPB = ROWS_GU / (WARPS * R);              // 20 row-blocks per expert

__global__ void gu_tiled2(const uint8_t* __restrict__ blob, const uint8_t* __restrict__ x,
                          const int* __restrict__ hx, const int32_t* __restrict__ begin,
                          const int32_t* __restrict__ hit_tok, const int32_t* __restrict__ hit_id, int n_hits,
                          float* __restrict__ gate_up) {
    const int e = (int) (blockIdx.x / RPB);
    const int i0 = (int) (blockIdx.x % RPB) * (WARPS * R) + (int) (threadIdx.x >> 5) * R;
    const int lane = threadIdx.x & 31;
    const int b0 = begin[e], b1 = begin[e + 1];
    if (b0 >= b1) return;
    const uint8_t* eb = blob + (size_t) e * BLOB;
    const uint8_t* codes = eb + O_GU_CODES + (size_t) i0 * ROW_GU;      // R rows, ROW_GU apart
    const uint8_t* scales = eb + O_GU_SCALES + (size_t) i0 * SC_GU * 2;

    for (int base = b0; base < b1; base += TM2) {
        const int m = min(TM2, b1 - base);
        float acc[R][TM2];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < TM2; ++t) acc[r][t] = 0.0f;

        int32_t hid[TM2];
        const uint8_t* xp[TM2];
        const int* hp[TM2];
        for (int t = 0; t < m; ++t) {
            hid[t] = hit_id[base + t];
            xp[t] = x + (size_t) hit_tok[base + t] * XROW;
            hp[t] = hx + (size_t) hit_tok[base + t] * NCH;
        }
        for (int c = lane; c < NCH; c += 32) {
            int cw[R][8];
            float dw[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t* cb = codes + (size_t) r * ROW_GU + (size_t) c * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned b = cb[j];
                    cw[r][j] = (int) ((b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) |
                                      (((b >> 6) & 3u) << 24));
                }
                dw[r] = f16_at(scales + (size_t) r * SC_GU * 2 + (size_t) (c >> 1) * 2);
            }
#pragma unroll
            for (int t = 0; t < TM2; ++t) {
                if (t >= m) break;
                const uint8_t* xb = xp[t] + (size_t) c * 34;
                const float dx = f16_at(xb);
                const int8_t* xq = (const int8_t*) (xb + 2);
                int xw[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) memcpy(&xw[j], xq + 4 * j, 4);
                const int hxc = hp[t][c];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int s = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j) s = __dp4a(cw[r][j], xw[j], s);
                    acc[r][t] += dw[r] * dx * (float) (s - hxc);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = i0 + r;
#pragma unroll
            for (int t = 0; t < TM2; ++t) {
                if (t >= m) break;
                const float s = warp_sum(acc[r][t]);
                if (lane != 0) continue;
                const int h = hid[t];
                const size_t o = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
                gate_up[o + (size_t) (i >> 1)] = s;
            }
        }
    }
}

// ------------------------------------------------- v3: remove the LOAD INSTRUCTIONS themselves
// R=8 gave 2.64x by spending one activation on 8 rows.  Both operands still cost absurd amounts to fetch:
//   * the codes are read as EIGHT single-byte loads per (row, chunk).  The row's 8 bytes are contiguous and -
//     because O_GU_CODES, ROW_GU=640 and the cudaMalloc base are all multiples of 8 - also 8-byte ALIGNED, so
//     one 8-byte load replaces eight byte-loads;
//   * the activation is a block_q8_0: 2-byte fp16 scale then 32 int8 on a 34-byte stride, so every read is
//     misaligned and costs 8 four-byte memcpys.  Stripping the scales out once per token (2.7 KB/token, one
//     pass) leaves a 32-byte-aligned pure-int8 array needing TWO uint4 loads instead of eight loads.
// The dp4a operand VALUES are unchanged and `hx` stays an int, so this is still bit-exact.
__global__ void strip_kernel(const uint8_t* __restrict__ x, int n_tok, uint8_t* __restrict__ xq,
                             __half* __restrict__ xs) {
    const int t = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (t >= n_tok) return;
    const int lane = threadIdx.x & 31;
    const uint8_t* xb = x + (size_t) t * XROW;
    uint8_t* dq = xq + (size_t) t * (NCH * 32);
    for (int c = lane; c < NCH; c += 32) {
        const uint8_t* s = xb + (size_t) c * 34;
        uint4* d = (uint4*) (dq + (size_t) c * 32);
        uint4 a, b;
        memcpy(&a, s + 2, 16); memcpy(&b, s + 18, 16);      // source unaligned, dst aligned
        d[0] = a; d[1] = b;
        ((__half*) (xs + (size_t) t * NCH))[c] = *(const __half*) s;
    }
}

template <int RR, int TT>
__global__ void gu_fast(const uint8_t* __restrict__ blob, const uint8_t* __restrict__ xq,
                        const __half* __restrict__ xs, const int* __restrict__ hx,
                        const int32_t* __restrict__ begin, const int32_t* __restrict__ hit_tok,
                        const int32_t* __restrict__ hit_id, int n_hits, float* __restrict__ gate_up) {
    constexpr int RPBX = ROWS_GU / (WARPS * RR);
    const int e = (int) (blockIdx.x / RPBX);
    const int i0 = (int) (blockIdx.x % RPBX) * (WARPS * RR) + (int) (threadIdx.x >> 5) * RR;
    const int lane = threadIdx.x & 31;
    const int b0 = begin[e], b1 = begin[e + 1];
    if (b0 >= b1) return;
    const uint8_t* eb = blob + (size_t) e * BLOB;
    const uint8_t* codes = eb + O_GU_CODES + (size_t) i0 * ROW_GU;
    const __half* scales = (const __half*) (eb + O_GU_SCALES) + (size_t) i0 * SC_GU;

    for (int base = b0; base < b1; base += TT) {
        const int m = min(TT, b1 - base);
        float acc[RR][TT];
#pragma unroll
        for (int r = 0; r < RR; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t) acc[r][t] = 0.0f;

        int32_t hid[TT];
        const uint8_t* xp[TT];
        const int* hp[TT];
        const __half* sp[TT];
        for (int t = 0; t < m; ++t) {
            hid[t] = hit_id[base + t];
            const int32_t tk = hit_tok[base + t];
            xp[t] = xq + (size_t) tk * (NCH * 32);
            hp[t] = hx + (size_t) tk * NCH;
            sp[t] = xs + (size_t) tk * NCH;
        }
        for (int c = lane; c < NCH; c += 32) {
            int cw[RR][8];
            float dw[RR];
#pragma unroll
            for (int r = 0; r < RR; ++r) {
                uint64_t raw = *(const uint64_t*) (codes + (size_t) r * ROW_GU + (size_t) c * 8);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned b = (unsigned) ((raw >> (8 * j)) & 0xffu);
                    cw[r][j] = (int) ((b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) |
                                      (((b >> 6) & 3u) << 24));
                }
                dw[r] = __half2float(scales[(size_t) r * SC_GU + (size_t) (c >> 1)]);
            }
            // 32 int8 per chunk = TWO uint4.  Reading eight uint32 out of a single uint4 would run 16 bytes
            // past it, so the pair is held as one 32-byte object and read through as words.
            struct X32 { uint4 a, b; };
            X32 xv[TT];
            for (int t = 0; t < m; ++t) {
                const uint4* src = (const uint4*) (xp[t] + (size_t) c * 32);
                xv[t].a = src[0]; xv[t].b = src[1];
            }
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t >= m) break;
                const float dx = __half2float(sp[t][c]);
                const int hxc = hp[t][c];
                const uint32_t* xw = (const uint32_t*) &xv[t];
#pragma unroll
                for (int r = 0; r < RR; ++r) {
                    int s = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j) s = __dp4a(cw[r][j], (int) xw[j], s);
                    acc[r][t] += dw[r] * dx * (float) (s - hxc);
                }
            }

        }
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            const int i = i0 + r;
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t >= m) break;
                const float s = warp_sum(acc[r][t]);
                if (lane != 0) continue;
                const int h = hid[t];
                const size_t o = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
                gate_up[o + (size_t) (i >> 1)] = s;
            }
        }
    }
}



static double ms_since(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    return (double) ms;
}

int main(int argc, char** argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 8192;          // tokens in the window
    const int E = argc > 2 ? atoi(argv[2]) : 512;           // experts per layer
    const int n_hits = C * TOPK;
    printf("window %d tokens x top-%d = %d hits over %d experts  (blob stride %.2f MiB)\n", C, TOPK, n_hits, E,
           BLOB / 1048576.0);

    std::vector<uint8_t> h_blob((size_t) E * BLOB);
    for (size_t i = 0; i < h_blob.size(); ++i) h_blob[i] = (uint8_t) ((i * 1103515245u) >> 13);
    std::vector<uint8_t> h_x((size_t) C * XROW);
    for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = (uint8_t) ((i * 2654435761u) >> 19);

    std::vector<int32_t> slot(n_hits), tok(n_hits);
    unsigned sd = 12345;
    for (int h = 0; h < n_hits; ++h) {
        sd = sd * 1103515245u + 12345u;
        slot[h] = (int32_t) ((sd >> 8) % (unsigned) E);
        tok[h] = h / TOPK;
    }
    // counting sort of the hits by expert - the plumbing the engine does not have yet
    std::vector<int32_t> cnt(E + 1, 0);
    for (int h = 0; h < n_hits; ++h) ++cnt[slot[h] + 1];
    for (int e = 0; e < E; ++e) cnt[e + 1] += cnt[e];
    std::vector<int32_t> cur = cnt, hit_tok(n_hits), hit_id(n_hits);
    for (int h = 0; h < n_hits; ++h) {
        const int p = cur[slot[h]]++;
        hit_tok[p] = tok[h];
        hit_id[p] = h;
    }
    int maxm = 0, used = 0;
    for (int e = 0; e < E; ++e) {
        maxm = std::max(maxm, cnt[e + 1] - cnt[e]);
        used += cnt[e + 1] > cnt[e];
    }
    printf("tokens per expert: mean %.1f  max %d  experts used %d/%d\n", (double) n_hits / E, maxm, used, E);

    uint8_t *d_blob, *d_x; int32_t *d_slot, *d_tok, *d_begin, *d_hit_tok, *d_hit_id; int* d_hx;
    float *d_a, *d_b;
    CK(cudaMalloc(&d_blob, h_blob.size()));
    CK(cudaMalloc(&d_x, h_x.size()));
    CK(cudaMalloc(&d_slot, n_hits * 4)); CK(cudaMalloc(&d_tok, n_hits * 4));
    CK(cudaMalloc(&d_begin, (E + 1) * 4));
    CK(cudaMalloc(&d_hit_tok, n_hits * 4)); CK(cudaMalloc(&d_hit_id, n_hits * 4));
    CK(cudaMalloc(&d_hx, (size_t) C * NCH * 4));
    const size_t ob = (size_t) n_hits * ROWS_GU * 4;
    CK(cudaMalloc(&d_a, ob)); CK(cudaMalloc(&d_b, ob));
    CK(cudaMemcpy(d_blob, h_blob.data(), h_blob.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, h_x.data(), h_x.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_slot, slot.data(), n_hits * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tok, tok.data(), n_hits * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_begin, cnt.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_hit_tok, hit_tok.data(), n_hits * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_hit_id, hit_id.data(), n_hits * 4, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ra(ob / 4), rb(ob / 4);
    const int reps = 5;
    const double wbytes = (double) n_hits * ROWS_GU * ROW_GU;   // what the GEMV actually walks

    {   // reference: one warp per (token-hit, row)
        const long long rows = (long long) n_hits * ROWS_GU;
        const unsigned blocks = (unsigned) ((rows + WARPS - 1) / WARPS);
        gu_ref<<<blocks, THREADS>>>(d_blob, d_slot, d_x, d_tok, n_hits, d_a);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) gu_ref<<<blocks, THREADS>>>(d_blob, d_slot, d_x, d_tok, n_hits, d_a);
        cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    }
    const double t_ref = ms_since(e0, e1) / reps;

    {   // tiled: one warp per (expert, row), looping over that expert's tokens
        const unsigned blocks = (unsigned) ((size_t) E * ROWS_PER_BLOCK);
        const unsigned hb = (unsigned) ((C + WARPS - 1) / WARPS);
        hx_kernel<<<hb, THREADS>>>(d_x, C, d_hx);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) {
            hx_kernel<<<hb, THREADS>>>(d_x, C, d_hx);
            gu_tiled<<<blocks, THREADS>>>(d_blob, d_x, d_hx, d_begin, d_hit_tok, d_hit_id, n_hits, d_b);
        }
        cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    }
    const double t_tile = ms_since(e0, e1) / reps;

    double t2 = 0.0;
    {   // v2: one warp per (expert, R rows), one activation chunk spent on R rows
        const unsigned blocks = (unsigned) ((size_t) E * RPB);
        CK(cudaMemset(d_b, 0, ob));
        gu_tiled2<<<blocks, THREADS>>>(d_blob, d_x, d_hx, d_begin, d_hit_tok, d_hit_id, n_hits, d_b);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r)
            gu_tiled2<<<blocks, THREADS>>>(d_blob, d_x, d_hx, d_begin, d_hit_tok, d_hit_id, n_hits, d_b);
        cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    }
    t2 = ms_since(e0, e1) / reps;

    // v3: aligned/stripped operands, one 8-byte code load per row-chunk, two uint4 loads per token-chunk
    uint8_t* d_xq; __half* d_xs;
    CK(cudaMalloc(&d_xq, (size_t) C * NCH * 32));
    CK(cudaMalloc(&d_xs, (size_t) C * NCH * 2));
    const unsigned hb2 = (unsigned) ((C + WARPS - 1) / WARPS);
    const double t_strip = [&] {
        strip_kernel<<<hb2, THREADS>>>(d_x, C, d_xq, d_xs);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) strip_kernel<<<hb2, THREADS>>>(d_x, C, d_xq, d_xs);
        cudaEventRecord(e1); CK(cudaDeviceSynchronize());
        return ms_since(e0, e1) / reps;
    }();

    auto bench_fast = [&](auto kern, int rpb) {
        const unsigned blocks = (unsigned) ((size_t) E * rpb);
        CK(cudaMemset(d_b, 0, ob));
        kern<<<blocks, THREADS>>>(d_blob, d_xq, d_xs, d_hx, d_begin, d_hit_tok, d_hit_id, n_hits, d_b);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r)
            kern<<<blocks, THREADS>>>(d_blob, d_xq, d_xs, d_hx, d_begin, d_hit_tok, d_hit_id, n_hits, d_b);
        cudaEventRecord(e1); CK(cudaDeviceSynchronize());
        return ms_since(e0, e1) / reps;
    };
    const double t_f48 = bench_fast(gu_fast<4, 8>, ROWS_GU / (WARPS * 4));
    const double t_f84 = bench_fast(gu_fast<8, 4>, ROWS_GU / (WARPS * 8));
    const double t_f88 = bench_fast(gu_fast<8, 8>, ROWS_GU / (WARPS * 8));   // last: parity below checks it




    CK(cudaMemcpy(ra.data(), d_a, ob, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rb.data(), d_b, ob, cudaMemcpyDeviceToHost));
    double worst = 0.0; long nbad = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
        const double d = std::fabs((double) ra[i] - (double) rb[i]);
        if (d > 0.0) { ++nbad; if (d > worst) worst = d; }
    }

    printf("\nref     (GEMV)  %8.2f ms  %7.0f tok/s   weight bytes walked %.2f GB\n", t_ref, 1e3 / t_ref * C,
           wbytes / 1e9);
    printf("tiled  (group)%8.2f ms  %7.0f tok/s   weight bytes walked %.2f GB\n", t_tile, 1e3 / t_tile * C,
           (double) used * ROWS_GU * ROW_GU / 1e9);
    printf("tiled2 (R=%d,T=4) %7.2f ms  %7.0f tok/s   <- activation amortized over %d rows\n", R, t2,
           1e3 / t2 * C, R);
    printf("strip  (once)  %8.3f ms  (cost of the aligned-activation pass)\n", t_strip);
    printf("fast R4 T8     %8.2f ms  %7.0f tok/s\n", t_f48, 1e3 / t_f48 * C);
    printf("fast R8 T4     %8.2f ms  %7.0f tok/s\n", t_f84, 1e3 / t_f84 * C);
    printf("fast R8 T8     %8.2f ms  %7.0f tok/s\n", t_f88, 1e3 / t_f88 * C);
    printf("SPEEDUP  tiled %.2fx  tiled2 %.2fx  fast(R8T8) %.2fx  fast+strip %.2fx\n",
           t_ref / t_tile, t_ref / t2, t_ref / t_f88, t_ref / (t_f88 + t_strip));
    printf("parity (fast R8T8 vs ref): %ld of %zu outputs differ, worst abs diff %.3e\n", nbad, ra.size(), worst);


    return 0;
}




