// include/strata/kernels/kv_q4.hpp - Q4_0 KV storage with Walsh-Hadamard rotation for QSA layers.
#pragma once

#include "strata/kernels/qsa.hpp"
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::kernels {

inline constexpr int QK4_0 = 32;

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;             // delta (half fp16 scale)
    uint8_t qs[QK4_0 / 2];  // 16 bytes = 32 4-bit nibbles
};
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

/// Bytes per cell (one token, one KV head): 8 blocks of 32 for head_dim=256 -> 144 bytes per head.
inline uint64_t kv_q4_bytes_per_head(int head_dim) {
    return (uint64_t) (head_dim / QK4_0) * sizeof(block_q4_0);
}

inline uint64_t kv_q4_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_q4_bytes_per_head((int) s.head_dim) * 2; // K and V
}

/// Orthonormal Fast Walsh-Hadamard Transform (FWHT) for dimension 256.
/// Self-inverse: H * H * x = x. Scale is 1 / sqrt(256) = 1/16.
void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream);

/// In-place FWHT256
inline void fwht256_inplace_cuda(float* data, int64_t n_rows, void* stream) {
    fwht256_cuda(data, data, n_rows, stream);
}

/// Append the cell at step[kStepPos] into Q4_0 pool with Hadamard rotation.
void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur,
                       const QsaShapes& s, void* stream);

/// Batched append of T cells into Q4_0 pool.
void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T,
                  const float* K, const float* V, const QsaShapes& s, void* stream);

/// Gather step[kStepWidth] cells named by `ids` from Q4_0 pool into FP16 scratch [id][kv_head][head_dim].
void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
