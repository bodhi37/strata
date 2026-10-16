// plan v0.4 P8c: the engine's record of the committed token stream.
//
// `SessionRecord` is the serve loop's host-side, append-only log of the exact token ids whose
// session state (QSA KV + indexer, GDN recurrence/conv, PLE history, ple_prev) is resident.
// It is appended ONLY where state actually advances (suffix prefill, Verifier::commit, the
// end-of-turn fix-up window) and cleared exactly where `session_zero` runs, so its length is
// by construction the true committed length that `sess_len` reports — and every request the
// gate re-verifies the proposed stream against it byte-for-byte, so a cache hit can never
// rest on a length alone.
//
// Host memory only: sized once at startup, never allocated on the token path, no device
// buffers, no CUDA-graph interaction.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::program {

class SessionRecord {
  public:
    void reserve(int64_t n) { ids_.reserve((size_t) n); }

    void reset() { ids_.clear(); }

    // Append `k` committed tokens. `t` may point into the caller's own record buffer region —
    // the reserve at startup makes the push_backs allocation-free in practice, but to stay
    // safe against pointer invalidation when t aliases ids_, append by value.
    void append(const int64_t* t, int64_t k) {
        for (int64_t i = 0; i < k; ++i) ids_.push_back((int32_t) t[i]);
    }
    void append(const int32_t* t, int64_t k) {
        for (int64_t i = 0; i < k; ++i) ids_.push_back(t[i]);
    }
    void append(int32_t t) { ids_.push_back(t); }

    int64_t len() const { return (int64_t) ids_.size(); }

    // True iff this record is a strict prefix of the proposed stream `ids[0..n)` — i.e. the
    // resident state's token stream matches the proposal token-for-token up to its length.
    // The caller still requires len() > 0 and len() < n (extend-only: the GDN/PLE state cannot
    // rewind, and the last prompt token must be new for the verify window).
    bool is_prefix_of(const int64_t* ids, int64_t n) const {
        if (len() > n) return false;
        for (int64_t i = 0; i < len(); ++i)
            if (ids_[(size_t) i] != (int32_t) ids[(size_t) i]) return false;
        return true;
    }

    const int32_t* data() const { return ids_.data(); }

  private:
    std::vector<int32_t> ids_;
};

}  // namespace strata::program
