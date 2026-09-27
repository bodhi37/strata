// src/kernels/cuda/kv_q4.cu - Q4_0 KV storage with Fast Walsh-Hadamard Transform
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fast Walsh-Hadamard Transform for N = 256
// 1 warp of 32 threads computes FWHT on 256 floats in registers
// Scale = 1 / sqrt(256) = 1/16 = 0.0625f
__global__ void fwht256_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n_rows, float scale) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size; // 8

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;

    const float* row_src = src + r * N;
    float* row_dst = dst + r * N;

    float reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = row_src[i * warp_size + lane] * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        row_dst[i * warp_size + lane] = reg[i];
    }
}

// 1 block = 1 32-value group of one KV head of K (blockIdx.z == 0) or V (1); 32 threads, 1 value each
__global__ void kv_append_q4_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    const float* __restrict__ kcur, const float* __restrict__ vcur,
                                    int kv_heads, int head_dim, int page_size) {
    const long long pos = (long long) __ldg(step + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const bool is_v = blockIdx.z == 1;
    const int blocks_per_head = head_dim / QK4_0; // 8 for 256
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0); // 144
    const float x = (is_v ? vcur : kcur)[h * head_dim + b * QK4_0 + t];

    float amax = fabsf(x);
    float max_val = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, max_val, o);
        if (a > amax) {
            amax = a;
            max_val = v;
        }
    }

    const float d = max_val / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t d_f16 = f16_from_f32(d);

    const float x_scaled = x * id;
    int q = __float2int_rz(x_scaled + 8.5f);
    uint8_t q_clamped = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));

    const uint8_t q_high = __shfl_down_sync(0xffffffffu, q_clamped, 16);

    const long long page = (long long) table[pos / page_size];
    const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
    uint8_t* base = (is_v ? v_q4 : k_q4) + row * bytes_per_head;
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(base) + b;

    if (t == 0) {
        blk->d = d_f16;
    }
    if (t < 16) {
        blk->qs[t] = q_clamped | (q_high << 4);
    }
}

// Gather step[kStepWidth] cells into FP16 scratch
__global__ void kv_gather_q4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0; // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0); // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x; // 0..31 inside block
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
    const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;

    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);

    const int j = t < 16 ? t : (t - 16);
    const uint8_t k_byte = k_blk->qs[j];
    const uint8_t v_byte = v_blk->qs[j];

    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);

    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    dim3 grid((unsigned) num_blocks);
    dim3 block(32, rows_per_block);
    const float scale = 1.0f / 16.0f; // 1 / sqrt(256)
    fwht256_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(src, dst, n_rows, scale);
    check("fwht256 launch");
}

// Batched append for prefill: grid = dim3(T, kv_heads, blocks_per_head), block = 32
__global__ void kv_append_q4_batch_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                          const int32_t* __restrict__ table, int64_t pos0,
                                          const float* __restrict__ K, const float* __restrict__ V,
                                          int kv_heads, int head_dim, int page_size, int is_v_grid) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const int blocks_per_head = head_dim / QK4_0; // 8 for 256
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0); // 144
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];

    float amax = fabsf(x);
    float max_val = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, max_val, o);
        if (a > amax) {
            amax = a;
            max_val = v;
        }
    }

    const float d = max_val / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t d_f16 = f16_from_f32(d);

    const float x_scaled = x * id;
    int q = __float2int_rz(x_scaled + 8.5f);
    uint8_t q_clamped = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));

    const uint8_t q_high = __shfl_down_sync(0xffffffffu, q_clamped, 16);

    const long long page = (long long) table[pos / page_size];
    const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
    uint8_t* base = (is_v ? v_q4 : k_q4) + row * bytes_per_head;
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(base) + b;

    if (th == 0) {
        blk->d = d_f16;
    }
    if (th < 16) {
        blk->qs[th] = q_clamped | (q_high << 4);
    }
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur,
                       const QsaShapes& s, void* stream) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "kv_append_q4: head_dim must be 256\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0), 2);
    kv_append_q4_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size);
    check("kv_append_q4 launch");
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T,
                  const float* K, const float* V, const QsaShapes& s, void* stream) {
    if (T <= 0) return;
    if (s.head_dim != 256) {
        std::fprintf(stderr, "kv_append_q4: head_dim must be 256\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0));
    cudaStream_t cs = (cudaStream_t) stream;
    kv_append_q4_batch_kernel<<<grid, 32, 0, cs>>>(
        k_q4, v_q4, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, 0);
    kv_append_q4_batch_kernel<<<grid, 32, 0, cs>>>(
        k_q4, v_q4, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, 1);
    check("kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    dim3 grid(num_blocks);
    dim3 block(32, rows_per_block);
    kv_gather_q4_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size,
        k_scratch, v_scratch);
    check("kv_gather_q4 launch");
}

}  // namespace strata::kernels
