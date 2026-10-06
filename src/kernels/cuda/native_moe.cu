// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_moe.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__global__ void __launch_bounds__(256)
combine(const float* __restrict__ parts, const float* __restrict__ weights,
        const float* __restrict__ shared, float* __restrict__ output, int64_t n_embd, int k) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    // Broadcast weights once per BLOCK via shared (was: every thread __ldg'd k
    // weights => 640 thr * 10 = 6400 loads of the same 10 floats).
    __shared__ float sw[15];
    if (threadIdx.x < k) sw[threadIdx.x] = __ldg(&weights[threadIdx.x]);
    __syncthreads();
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    float sum = __ldg(&parts[col]) * sw[0];
    for (int expert = 1; expert < k; ++expert) {
        sum = fmaf(__ldg(&parts[int64_t(expert) * n_embd + col]), sw[expert], sum);
    }
    if (shared) sum += __ldg(&shared[col]);
    output[col] = sum;
}
// float4-vectorized combine for the common n_embd % 4 == 0 shapes (e.g. 2560): one thread covers
// 4 columns, quartering the weight-broadcast and index overhead.  Per-element expression is
// identical to `combine`, so the numerics match exactly.
__global__ void __launch_bounds__(256)
combine_vec4(const float* __restrict__ parts, const float* __restrict__ weights,
             const float* __restrict__ shared, float* __restrict__ output, int64_t n_embd, int k) {
    // True LDG.128 path for n_embd%4==0: one float4 per expert per thread.
    // Old version was scalar x4 (misnamed vec4) with per-thread weight loads.
    __shared__ float sw[15];
    if (threadIdx.x < k) sw[threadIdx.x] = __ldg(&weights[threadIdx.x]);
    __syncthreads();
    const int64_t col4 = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) return;
    const int tail = (col4 + 4 <= n_embd) ? 4 : (int) (n_embd - col4);
    if (tail == 4) {
        // NOTE: no float4*float operator in CUDA (helper_math.h is not included),
        // so the broadcast multiply is written component-wise. Same values.
        const float4 p0 = __ldg(reinterpret_cast<const float4*>(&parts[col4]));
        float4 acc = make_float4(p0.x * sw[0], p0.y * sw[0], p0.z * sw[0], p0.w * sw[0]);
        for (int expert = 1; expert < k; ++expert) {
            const float4 pv = __ldg(reinterpret_cast<const float4*>(&parts[int64_t(expert) * n_embd + col4]));
            acc.x = fmaf(pv.x, sw[expert], acc.x);
            acc.y = fmaf(pv.y, sw[expert], acc.y);
            acc.z = fmaf(pv.z, sw[expert], acc.z);
            acc.w = fmaf(pv.w, sw[expert], acc.w);
        }
        if (shared) {
            float4 sv = __ldg(reinterpret_cast<const float4*>(&shared[col4]));
            acc.x += sv.x; acc.y += sv.y; acc.z += sv.z; acc.w += sv.w;
        }
        *reinterpret_cast<float4*>(&output[col4]) = acc;
    } else {
        for (int c = 0; c < tail; ++c) {
            const int64_t col = col4 + c;
            float sum = __ldg(&parts[col]) * sw[0];
            for (int expert = 1; expert < k; ++expert) {
                sum = fmaf(__ldg(&parts[int64_t(expert) * n_embd + col]), sw[expert], sum);
            }
            if (shared) sum += __ldg(&shared[col]);
            output[col] = sum;
        }
    }
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    // LDG.128 needs 16B-aligned rows: n_embd%4==0 keeps every expert row at a
    // 16B stride, but the BASE must be 16B too (valid_span only promises 4B).
    // cudaMalloc/arena bases always are; the scalar path covers the rest.
    const auto aligned16 = [](const void* q) { return (reinterpret_cast<uintptr_t>(q) & 15u) == 0; };
    if ((n_embd & 3) == 0 && aligned16(parts) && aligned16(output) && (shared == nullptr || aligned16(shared))) {
        combine_vec4<<<unsigned(((n_embd / 4) + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            parts, weights, shared, output, n_embd, int(k));
    } else {
        combine<<<unsigned((n_embd + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            parts, weights, shared, output, n_embd, int(k));
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    combine<<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
