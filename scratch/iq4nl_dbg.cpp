#include "strata/kernels/cpu/iq_avx512.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <vector>
namespace c = strata::kernels::cpu;
int main() {
    ggml_cpu_init();
    uint8_t wblk[18];
    const uint16_t d = 0x3400;  // fp16
    std::memcpy(wblk, &d, 2);
    for (int i = 0; i < 16; ++i) wblk[2 + i] = (uint8_t) i;   // lo nibble = i, hi nibble = 0
    uint8_t yblk[34];
    const uint16_t dy = 0x3c00;  // fp16 1.0
    std::memcpy(yblk, &dy, 2);
    for (int i = 0; i < 32; ++i) yblk[2 + i] = (int8_t)(i - 16);
    float mine = -1e30f;
    float* mp[1] = { &mine };
    const void* hp[1] = { yblk };     // PROPER pointer array
    c::iq4nl_rows_multi(wblk, 18, 32, hp, 1, mp, 0, 1);
    const auto* td = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    float ref = 0;
    td->vec_dot(32, &ref, 0, wblk, 0, yblk, 0, 1);
    double sref = 0;
    const double dd = 0x3400;  // placeholder, computed below
    double dxs = 1.0;          // fp16 0x3400 = 0.25? compute:
    // fp16 0x3400: exp bits = (0x3400>>10)&31 = 13 -> 2^(13-15)=0.25, mantissa 0 -> 0.25
    dxs = 0.25; const double dys = 1.0;
    for (int j = 0; j < 16; ++j) {
        sref += (double) (int8_t) yblk[2 + j] * kvalues_iq4nl[wblk[2 + j] & 0xf];
        sref += (double) (int8_t) yblk[2 + 16 + j] * kvalues_iq4nl[wblk[2 + j] >> 4];
    }
    sref *= dxs * dys;
    std::printf("mine %.6f  ggml %.6f  scalar %.6f\n", mine, ref, sref);
    return 0;
}
