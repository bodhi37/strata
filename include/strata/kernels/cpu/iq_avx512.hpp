// include/strata/kernels/cpu/iq_avx512.hpp - plan v0.3 P6: AVX-512 multi-token dot products for the i-quant
// expert formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) against Q8_K activations (ggml's block_q8_K).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool iq512_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// R10: multi-token AVX-512 IQ4_NL rows against Q8_0 activations (the down projection of every native
/// pack here).  `w` points at the first down row, `row_bytes` is ggml_row_size(IQ4_NL, n), `hq[t]` are
/// n-value Q8_0 buffers.  One block decodes once; each token is one load/sign/maddubs/madd.  The
/// per-block integers equal ggml's kernel exactly; only the float addition order differs (rel ~1e-7).
void iq4nl_rows_multi(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
