// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/platform/memory.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <filesystem>
#include <sstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // std::numeric_limits<T>::max() below
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// a 64-bit seek (as in pinned.cu): the 32-bit `fseek` wraps past 4 GiB, and the spelling differs per platform
#ifndef STRATA_FSEEK64
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif
#endif

namespace strata::core {

namespace detail {

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's inactive_file can race memory.current, so bound it to charged usage first.
    uint64_t reclaimable = std::min(stat.inactive_file, stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from) {
    if (base_bytes > budget) return -1;
    const int64_t slots = (int64_t) slot_bytes.size();
    if (lend_from < 0 || lend_from > slots) lend_from = slots;   // no lend region: only the experts no slot holds
    int64_t keep = slots;
    uint64_t bytes = base_bytes;
    while (keep > lend_from) {
        const uint64_t b = slot_bytes[(size_t) keep - 1];
        if (b > budget - bytes) break;
        bytes += b;
        --keep;
    }
    return keep;
}

bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out) {
    if (in == out || in >= offsets.size() || out >= offsets.size() || offsets[in] == kNoCacheComplement ||
        offsets[out] != kNoCacheComplement) return false;
    offsets[out] = offsets[in];
    offsets[in] = kNoCacheComplement;
    return true;
}

}  // namespace detail

namespace {

#if defined(__linux__)
bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;

    bool inactive_file = false, file_dirty = false, file_writeback = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        if (key == "inactive_file") {
            if (inactive_file) return false;
            inactive_file = true;
            stat.inactive_file = value;
        } else if (key == "file_dirty") {
            if (file_dirty) return false;
            file_dirty = true;
            stat.file_dirty = value;
        } else if (key == "file_writeback") {
            if (file_writeback) return false;
            file_writeback = true;
            stat.file_writeback = value;
        }
    }
    if (!input.eof() || !inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}
#endif

}  // namespace

namespace detail {

bool host_available_memory(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                           const std::string& cgroup_root) {
    m = HostMemory{};
#if defined(_WIN32)
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    m.available = (uint64_t) status.ullAvailPhys;
    return m.available > 0;
#elif defined(__linux__)
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info(meminfo);
    std::string line;
    uint64_t bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    std::ifstream groups(self_cgroup);
    if (!groups) return false;
    const std::filesystem::path root(cgroup_root);
    bool v2 = false;
    std::string v1_path;   // the memory controller's group (cgroup v1), when there is no v2 line
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) {
            // v1: "N:controller[,controller]:/path"
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
            for (std::string c; std::getline(ctl, c, ',');)
                if (c == "memory") v1_path = line.substr(c2 + 1);
            continue;
        }
        auto path = (root / line.substr(4)).lexically_normal();
        if (path.string().rfind(root.string(), 0) != 0 || !std::filesystem::is_directory(path)) return false;
        v2 = true;
        while (path.string().rfind(root.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            const bool readable = bool(limit_file >> limit) && bool(current_file >> current);
            // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
            if (!readable && !(path == root && !std::filesystem::exists(path / "memory.max") &&
                               std::filesystem::exists(path / "cgroup.controllers"))) return false;
            if (readable && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                    m.cgroup_limit = std::min(m.cgroup_limit, cap);
                } catch (...) { return false; }
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    if (!v2 && !v1_path.empty()) {
        // #633: cgroup v1 (older Docker hosts, RHEL 7/8): the memory controller's group and its ancestors.  An
        // unlimited group says a number near 2^63 (rounded to its page size); a group whose files are not
        // visible (no mount in this namespace) is skipped - MemAvailable alone, as with no cgroup at all.
        const std::filesystem::path mroot = root / "memory";
        auto path = (mroot / v1_path.substr(v1_path.rfind('/', 0) == 0 ? 1 : 0)).lexically_normal();
        while (path.string().rfind(mroot.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.limit_in_bytes"), usage_file(path / "memory.usage_in_bytes");
            uint64_t cap = 0, usage = 0;
            if (limit_file >> cap && usage_file >> usage && cap < (1ull << 62)) {
                bytes = std::min(bytes, usage < cap ? cap - usage : 0);
                m.cgroup_limit = std::min(m.cgroup_limit, cap);
            }
            if (path == mroot) break;
            path = path.parent_path();
        }
    }
    m.available = bytes;
    return true;
#else
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0 ||
        (uint64_t) pages > std::numeric_limits<uint64_t>::max() / (uint64_t) page_bytes) return false;
    m.available = (uint64_t) pages * (uint64_t) page_bytes;
    return m.available > 0;
#endif
}

}  // namespace detail

namespace {

bool available_memory_bytes(uint64_t& bytes) {
    detail::HostMemory m;
    if (!detail::host_available_memory(m)) return false;
    bytes = m.available;
    return bytes > 0;
}

}  // namespace

// ================================ THE FILE-BACKED SOURCE =========================
FileExpertSource::~FileExpertSource() { close(); }

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (layout.n_layers != n_layers || layout.n_expert != n_expert) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<int64_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }
    const uint64_t blob_count = (uint64_t) n_layers * (uint64_t) n_expert;
    if (blob_count > (uint64_t) std::numeric_limits<int64_t>::max() ||
        (uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }

    std::vector<uint64_t> layer_offsets((size_t) n_layers), layer_blob_bytes((size_t) n_layers);
    const uint64_t want = layout.total;
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the loaded expert layout has an invalid size";
        return false;
    }
    if (!layout.native) {
        if (blob_count > std::numeric_limits<uint64_t>::max() / (uint64_t) strata::kernels::cpu::BLOB) {
            err = "FileExpertSource: the canonical expert size overflows";
            return false;
        }
        const uint64_t canonical_size = blob_count * (uint64_t) strata::kernels::cpu::BLOB;
        if (want != canonical_size) {
            err = "FileExpertSource: the canonical expert layout has an inconsistent size";
            return false;
        }
        const uint64_t bytes = (uint64_t) strata::kernels::cpu::BLOB;
        const uint64_t layer_bytes = (uint64_t) n_expert * bytes;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            layer_offsets[(size_t) layer] = (uint64_t) layer * layer_bytes;
            layer_blob_bytes[(size_t) layer] = bytes;
        }
    } else {
        if (layout.offset.size() != (size_t) n_layers || layout.bytes.size() != (size_t) n_layers ||
            layout.fmt.size() != (size_t) n_layers) {
            err = "FileExpertSource: the native expert layout is incomplete";
            return false;
        }
        uint64_t at = 0;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            const size_t i = (size_t) layer;
            const uint64_t bytes = (uint64_t) layout.fmt[i].bytes;
            if (layout.offset[i] != at || bytes == 0 || layout.bytes[i] != bytes ||
                bytes > std::numeric_limits<uint64_t>::max() / (uint64_t) n_expert) {
                err = "FileExpertSource: the native expert layout is invalid at layer " + std::to_string(layer);
                return false;
            }
            const uint64_t layer_bytes = bytes * (uint64_t) n_expert;
            if (at > want || layer_bytes > want - at) {
                err = "FileExpertSource: the native expert layout exceeds its declared size at layer " +
                      std::to_string(layer);
                return false;
            }
            layer_offsets[i] = layout.offset[i];
            layer_blob_bytes[i] = bytes;
            at += layer_bytes;
        }
        if (at != want) {
            err = "FileExpertSource: the native expert layout has an inconsistent size";
            return false;
        }
    }
    const std::string path = pack_dir + "/experts.bin";
    if (layout.native && !gguf_.empty() && !std::filesystem::exists(path)) {
        // CS-T: no experts.bin - the model's GGUF shards, read in place
        blobs_ = (int64_t) blob_count;
        n_layers_ = n_layers;
        n_expert_ = n_expert;
        layer_offsets_ = std::move(layer_offsets);
        layer_blob_bytes_ = std::move(layer_blob_bytes);
        if (!open_gguf(err)) { close(); return false; }
        return true;
    }

#if defined(_WIN32)
    // UTF-8 -> UTF-16: the pack may live under a path with non-ASCII characters, and `CreateFileA` would
    // silently mangle it into a file-not-found.
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    // **`FILE_FLAG_RANDOM_ACCESS` WAS HERE AND IT COST 14x.**
    //
    // The design depends on the OS page cache holding the whole 34 GB expert set, because this machine has
    // 64 GB of DDR5 and `L9` measured the CPU path at 44.14 GB/s from DRAM.  `FILE_FLAG_RANDOM_ACCESS` tells
    // the cache manager the opposite: it disables read-ahead AND it lets the manager drop the pages again
    // quickly, on the assumption that a large randomly-accessed file will not be re-read.  Measured, on
    // `strata generate --max-new 24`: **1.93 GB/s** - disk speed, 344 ms/token, and it never warmed up over 25
    // tokens, because the pages were being evicted as fast as they were faulted in.
    //
    // The correct flag is NO flag.  The access pattern IS random (10 of 512 experts per layer, a different 10
    // each layer), but every byte read is read again on the next token, so retention is the whole game.
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = "FileExpertSource: cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        err = "FileExpertSource: cannot size " + path;
        return false;
    }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) sz.QuadPart, (unsigned long long) want);
        CloseHandle(f);
        err = buf;
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        err = "FileExpertSource: CreateFileMapping failed on " + path;
        return false;
    }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        err = "FileExpertSource: MapViewOfFile failed on " + path;
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
    paths_.assign(1, path);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if (st.st_size < 0 || (uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) (st.st_size < 0 ? 0 : st.st_size),
                      (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
#endif
    blobs_ = (int64_t) blob_count;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    mapped_bytes_ = want;
    layer_offsets_ = std::move(layer_offsets);
    layer_blob_bytes_ = std::move(layer_blob_bytes);
    return true;
}

void FileExpertSource::close() {
    if (complement_arena_ != nullptr) {
        if (complement_pinned_ && !complement_partial_) (void) cudaFreeHost(complement_arena_);
        else {
            if (complement_partial_) (void) cudaHostUnregister(complement_arena_);
            if (complement_locked_ > 0)
                strata::platform::unlock_resident((uint8_t*) complement_arena_ + complement_lock_off_, complement_locked_);
            std::free(complement_arena_);
        }
    }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
    }
    xstage_ = nullptr;
    xstage_pinned_ = false;
    xstage_cap_ = 0;
    xstage_blob_ = 0;
    override_.clear();
    staged_.clear();
    exchanges_ = 0;
    file_reads_.store(0);
    complement_arena_ = nullptr;
    complement_host_ = nullptr;
    complement_device_ = nullptr;
    complement_bytes_ = 0;
    complement_offsets_.clear();
    complement_pinned_ = false;
    complement_partial_ = false;
    complement_pin_limit_ = 0;
    complement_lock_off_ = 0;
    complement_ready_ = false;
    complement_locked_ = 0;
    complement_lent_slots_ = 0;
    if (!maps_.empty()) {
        for (Map& m : maps_) {
#if defined(_WIN32)
            if (m.base != nullptr) UnmapViewOfFile((LPCVOID) m.base);
            if (m.mapping != nullptr) CloseHandle((HANDLE) m.mapping);
            if (m.file != nullptr) CloseHandle((HANDLE) m.file);
#else
            if (m.base != nullptr) munmap((void*) m.base, (size_t) m.bytes);
            if (m.fd >= 0) ::close(m.fd);
#endif
        }
        maps_.clear();
        base_ = nullptr;   // one of the views above
    }
    role_ptr_.clear();
    role_bytes_.clear();
    role_file_.clear();
    paths_.clear();
#if defined(_WIN32)
    for (void* h : direct_) CloseHandle((HANDLE) h);
#endif
    direct_.clear();
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_buf_.clear();
        stage_key_.clear();
        stage_epoch_.clear();
        stage_used_.clear();
        stage_busy_.clear();
        stage_of_.clear();
        stage_blob_ = 0;
        stage_seq_ = 0;
        epoch_ = 0;
        last_layer_ = -1;
        stage_grew_ = false;
    }
    ram_reads_.store(0);
    warm_stamp_.reset();
    warm_hits_.store(0);
    warm_count_.store(0);
    file_read_bytes_.store(0);
    file_blob_bytes_.store(0);
    file_us_.store(0);
#if defined(_WIN32)
    if (base_ != nullptr) UnmapViewOfFile((LPCVOID) base_);
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (base_ != nullptr) munmap((void*) base_, (size_t) mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ = nullptr;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    mapped_bytes_ = 0;
    layer_offsets_.clear();
    layer_blob_bytes_.clear();
    reads_ = 0;
}


bool ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const uint8_t* b = blob(layer, expert);
    if (b == nullptr || dst == nullptr) return false;
    std::memcpy(dst, b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(layer));
    return true;
}

// ================================ CS-T: THE GGUF SHARDS IN PLACE =========================//
// A native pack without experts.bin: every file native_experts.txt names is mapped (MapViewOfFile / mmap, no
// flag - the same retention argument as experts.bin above), and an expert's blob [gate rows | up rows | down rows]
// is three slices of three tensors, possibly in two shards (UD-Q4_K_XL's layer 11).  Nothing is read at open; a
// blob is assembled when it is asked for (`blob`, into a small pool of buffers) or copied where it is needed
// (`copy_blob`: the RAM copy, the prompt path's pinned stager buffers).
bool FileExpertSource::open_gguf(std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!check_experts_gguf(gguf_, lay, err)) { err = "FileExpertSource: " + err; return false; }
    const size_t cut = gguf_.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf_.substr(0, cut + 1);
    std::map<std::string, size_t> index;
    role_ptr_.assign((size_t) (3 * n_layers_), nullptr);
    role_bytes_.assign((size_t) (3 * n_layers_), 0);
    role_file_.assign((size_t) (3 * n_layers_), 0);
    for (int64_t l = 0; l < n_layers_; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, lay.bytes[(size_t) l] - fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * l + r);
            const std::string path = lay.gguf_file.size() > i && !lay.gguf_file[i].empty() ? dir + lay.gguf_file[i]
                                                                                            : gguf_;
            auto it = index.find(path);
            if (it == index.end()) {
                Map m;
#if defined(_WIN32)
                const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
                std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
                if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
                HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL, nullptr);
                LARGE_INTEGER sz{};
                if (f == INVALID_HANDLE_VALUE || !GetFileSizeEx(f, &sz)) {
                    if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                HANDLE mh = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
                void* view = mh != nullptr ? MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
                if (view == nullptr) {
                    if (mh != nullptr) CloseHandle(mh);
                    CloseHandle(f);
                    err = "FileExpertSource: cannot map " + path;
                    return false;
                }
                m.file = f;
                m.mapping = mh;
                m.bytes = (uint64_t) sz.QuadPart;
                m.base = (const uint8_t*) view;
#else
                const int fd = ::open(path.c_str(), O_RDONLY);
                struct stat st{};
                if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
                    if (fd >= 0) ::close(fd);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                void* view = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: cannot map " + path; return false; }
                m.fd = fd;
                m.bytes = (uint64_t) st.st_size;
                m.base = (const uint8_t*) view;
#endif
                maps_.push_back(m);
                paths_.push_back(path);
                it = index.emplace(path, maps_.size() - 1).first;
            }
            const Map& m = maps_[it->second];
            role_file_[i] = (int) it->second;
            const uint64_t at = lay.gguf_off[i], bytes = per[r] * (uint64_t) n_expert_;
            if (at > m.bytes || bytes > m.bytes - at) {   // check_experts_gguf proved it; the mapping must agree
                err = "FileExpertSource: an expert span runs past the end of " + path;
                return false;
            }
            role_ptr_[i] = m.base + (size_t) at;
            role_bytes_[i] = per[r];
        }
    }
    base_ = maps_.front().base;       // "opened"; mapped_blob answers nullptr in this mode
    warm_stamp_.reset(new std::atomic<uint32_t>[(size_t) (n_layers_ * n_expert_)]());
    for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    return true;
}

bool FileExpertSource::copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const {
    if (dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    if (!direct_.empty()) {   // #286: from the drive; a failed read falls back to the mapping below
        const Fill f{0, layer, expert, dst};
        if (read_direct(&f, 1)) return true;
    }
    if (!role_ptr_.empty()) {
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint64_t per = role_bytes_[i];
            std::memcpy(dst + at, role_ptr_[i] + (size_t) ((uint64_t) expert * per), (size_t) per);
            at += per;
        }
        return true;
    }
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr) return false;
    std::memcpy(dst, b, (size_t) layer_blob_bytes_[(size_t) layer]);
    return true;
}

// A blob assembled from the three role slices.  The buffer of a (layer, expert) is reused for another only once
// its blob has not been asked for during `kStageAge` layers (begin_layer) or 256 assemblies, whichever comes
// first, and never while it is being filled: the pool computes a layer's misses before it starts the next, and a
// fill (the GPU cache at startup, an adaptive swap, a helper GPU) copies the blob right away.
//
// `claim_stage` finds or reserves the buffer of `key` (stage_mu_ held): true when the blob is already there (or
// being filled by another thread - the caller then waits), false when the caller must fill buffer `v`.
bool FileExpertSource::claim_stage(int64_t key, size_t& v, bool& fill) {
    constexpr uint64_t kStageSeq = 256;
    const uint64_t seq = ++stage_seq_;
    auto it = stage_of_.find(key);
    fill = false;
    if (it != stage_of_.end()) {
        v = it->second;
        stage_epoch_[v] = epoch_;
        stage_used_[v] = seq;
        return true;
    }
    v = stage_buf_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < stage_buf_.size(); ++i)
        if (!stage_busy_[i] && (stage_epoch_[i] + kStageAge <= epoch_ || stage_used_[i] + kStageSeq <= seq) &&
            stage_used_[i] < oldest) {
            oldest = stage_used_[i];
            v = i;
        }
    if (v == stage_buf_.size()) {
        stage_buf_.emplace_back(new (std::nothrow) uint8_t[(size_t) stage_blob_]);
        if (!stage_buf_.back()) { stage_buf_.pop_back(); return false; }
        stage_key_.push_back(-1);
        stage_epoch_.push_back(0);
        stage_used_.push_back(0);
        stage_busy_.push_back(0);
        if (stage_buf_.size() == 512 && !stage_grew_) {
            stage_grew_ = true;
            std::fprintf(stderr, "FileExpertSource: %zu blobs assembled from the GGUF are in use at once (%.2f GiB)\n",
                         stage_buf_.size(), (double) stage_buf_.size() * (double) stage_blob_ / 1073741824.0);
        }
    } else {
        stage_of_.erase(stage_key_[v]);
    }
    stage_key_[v] = key;
    stage_epoch_[v] = epoch_;
    stage_used_[v] = seq;
    stage_busy_[v] = 1;
    stage_of_[key] = v;
    fill = true;
    return false;
}

// Fills buffer `v` (reserved by claim_stage) outside the lock, then publishes it.
bool FileExpertSource::fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = copy_from_files(layer, expert, dst);
    publish_stage(v, layer, ok, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    return ok;
}

void FileExpertSource::publish_stage(size_t v, int64_t layer, bool ok, double us) {
    file_us_.fetch_add((uint64_t) us, std::memory_order_relaxed);
    if (ok) {
        file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        file_blob_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_busy_[v] = 0;
        if (!ok) {
            stage_of_.erase(stage_key_[v]);
            stage_key_[v] = -1;
        }
    }
    stage_cv_.notify_all();
}

const uint8_t* FileExpertSource::staged_blob(int64_t layer, int64_t expert) {
    const int64_t key = layer * n_expert_ + expert;
    size_t v = 0;
    bool fill = false;
    uint8_t* dst = nullptr;
    {
        std::unique_lock<std::mutex> lk(stage_mu_);
        const bool have = claim_stage(key, v, fill);
        if (!have && !fill) return nullptr;
        dst = stage_buf_[v].get();
        if (have) {
            // another thread (a prefetch, the adaptive tier) is filling it: wait for that
            stage_cv_.wait(lk, [&] { return !stage_busy_[v] || stage_key_[v] != key; });
            if (stage_key_[v] != key) return nullptr;   // its fill failed
            return dst;
        }
    }
    return fill_stage(v, layer, expert, dst) ? dst : nullptr;
}

void FileExpertSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    if (!staged() || n <= 0 || layer < 0 || layer >= n_layers_) return;
    std::vector<Fill> todo;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (complement_ready_ && index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement)
                continue;                                     // in the RAM copy
            if (!override_.empty() && override_[index] != nullptr) continue;
            size_t v = 0;
            bool fill = false;
            if (!claim_stage(layer * n_expert_ + e, v, fill) && fill) {
                todo.push_back({v, layer, e, stage_buf_[v].get()});
                if (warm_stamp_) {
                    const uint32_t s = warm_stamp_[index].load(std::memory_order_relaxed);
                    if (s != 0 && (uint64_t) s + 3 >= epoch_ + 1) warm_hits_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    fill_many(todo);
}

void FileExpertSource::prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) {
    if (direct_.empty() || pairs == nullptr || n <= 0) return;   // unbuffered only: the mapped fill stays as it was
    std::vector<Fill> todo;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (int64_t i = 0; i < std::min<int64_t>(n, 64); ++i) {
            const int64_t l = pairs[i].first, e = pairs[i].second;
            if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) continue;
            size_t v = 0;
            bool fill = false;
            if (!claim_stage(l * n_expert_ + e, v, fill) && fill) todo.push_back({v, l, e, stage_buf_[v].get()});
        }
    }
    fill_many(todo);
}

void FileExpertSource::fill_many(const std::vector<Fill>& todo) {
    if (todo.empty()) return;
    if (!direct_.empty()) {
        // #286: overlapped batches - every role window of 16 blobs in flight at once; a big batch (the profile
        // fill) on up to 4 threads, a decode layer's few misses on this one
        std::atomic<size_t> next{0};
        auto work = [&] {
            for (size_t at; (at = next.fetch_add(16)) < todo.size();) {
                const size_t k = std::min<size_t>(16, todo.size() - at);
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = read_direct(todo.data() + at, k);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                for (size_t i = at; i < at + k; ++i) {
                    const Fill& f = todo[i];
                    // a failed batch: each blob again on its own (copy_from_files falls back to the mapping)
                    if (ok) publish_stage(f.v, f.layer, true, us / (double) k);
                    else (void) fill_stage(f.v, f.layer, f.e, f.dst);
                }
            }
        };
        const size_t nt = std::min<size_t>(4, (todo.size() + 15) / 16);
        std::vector<std::thread> th;
        for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
        work();
        for (auto& t : th) t.join();
        return;
    }
#if defined(_WIN32)
    // One PrefetchVirtualMemory call for every slice about to be copied: the memory manager reads them in large
    // requests, all queued at once, where the copies' page faults would read a few clusters each.  The copies below
    // then find the pages resident (or in flight).  STRATA_FETCH_PVM=0 is the A/B arm.
    {
        using Pvm = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        static const Pvm pvm = [] {
            const char* v = std::getenv("STRATA_FETCH_PVM");
            if (v != nullptr && std::atoi(v) == 0) return (Pvm) nullptr;
            return (Pvm) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
        }();
        if (pvm != nullptr) {
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            ranges.reserve(todo.size() * 3);
            for (const Fill& f : todo)
                for (int r = 0; r < 3; ++r) {
                    const size_t i = (size_t) (3 * f.layer + r);
                    ranges.push_back({(PVOID) (role_ptr_[i] + (size_t) ((uint64_t) f.e * role_bytes_[i])),
                                      (SIZE_T) role_bytes_[i]});
                }
            (void) pvm(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
        }
    }
#endif
    // the page faults of a mapped read are one outstanding request each: several threads keep the SSD's queue full
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next.fetch_add(1)) < todo.size();)
            (void) fill_stage(todo[i].v, todo[i].layer, todo[i].e, todo[i].dst);
    };
    const size_t nt = std::min<size_t>(todo.size(), (size_t) fetch_threads_);
    std::vector<std::thread> th;
    for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
}

uint64_t FileExpertSource::expert_bytes() const {
    uint64_t total = 0;
    for (uint64_t b : layer_blob_bytes_) total += b * (uint64_t) n_expert_;
    return total;
}

bool FileExpertSource::open_direct(std::string& why) {
#if defined(_WIN32)
    for (const std::string& path : paths_) {
        const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> w((size_t) (wide > 0 ? wide : 1), L'\0');
        if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wide);
        HANDLE h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            why += "; cannot open " + path + " unbuffered (error " + std::to_string((unsigned long long) GetLastError()) +
                   "), read through the file cache";
            for (void* d : direct_) CloseHandle((HANDLE) d);
            direct_.clear();
            return false;
        }
        direct_.push_back(h);
    }
    if (role_ptr_.empty()) {   // experts.bin: blob() now assembles into the stage buffers, sized for the largest blob
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    }
    return true;
#else
    (void) why;
    return false;
#endif
}

bool FileExpertSource::set_unbuffered(uint64_t ram_bytes, std::string& why) {
#if defined(_WIN32)
    if (base_ == nullptr || paths_.empty() || !direct_.empty()) {
        why = !direct_.empty() ? "already unbuffered" : "no expert files open";
        return !direct_.empty();
    }
    // #577: the file tier reads the experts outside the RAM copy, not every byte of the shards (dense weights, the
    // PLE table), and the copy holds at most every expert - the budget asked for can be more than that
    const uint64_t experts = expert_bytes();
    const uint64_t arena = std::min(ram_bytes, experts);
    if (!experts_unbuffered(paths_, arena, why, /*cache_counts=*/false, experts - arena)) return false;
    return open_direct(why);
#else
    (void) ram_bytes;
    why = "through the file cache (not Windows)";
    return false;
#endif
}

bool FileExpertSource::recheck_unbuffered(std::string& why) {
#if defined(_WIN32)
    if (const char* env = std::getenv("STRATA_UNBUFFERED_LOAD"); env != nullptr && env[0] != '\0') {
        why = std::string("STRATA_UNBUFFERED_LOAD=") + env;
        return !direct_.empty();
    }
    if (base_ == nullptr || paths_.empty()) {
        why = "no expert files open";
        return false;
    }
    // #577: the RAM copy is built (and already out of the available RAM), so what the file cache would have to keep
    // is exactly the experts outside it - the GPU cache's (refilled after a prompt borrowed their slots) and the
    // ones neither holds
    const uint64_t experts = expert_bytes();
    const uint64_t read = experts > complement_bytes_ ? experts - complement_bytes_ : 0;
    std::string w;
    const bool ub = experts_unbuffered(paths_, 0, w, /*cache_counts=*/false, read);
    char head[96];
    std::snprintf(head, sizeof head, "re-checked with the RAM copy built (%.2f GiB): ",
                  (double) complement_bytes_ / 1073741824.0);
    why = head + w;
    if (ub == !direct_.empty()) return ub;
    if (ub) return open_direct(why);
    // through the file cache after all: the mapped reads take over (staged() stays true for the GGUF in place)
    for (void* d : direct_) CloseHandle((HANDLE) d);
    direct_.clear();
    return false;
#else
    why = "through the file cache (not Windows)";
    return false;
#endif
}

bool FileExpertSource::read_direct(const Fill* fills, size_t n) const {
#if defined(_WIN32)
    // NTFS runs the unbuffered reads of a file one at a time while the file is mapped or cached anywhere (the engine
    // maps the GGUF shards for the token embedding and the PLE table): measured on a PCIe 5 drive, 512 KiB reads at
    // queue depth 48 make 10.4 GB/s on a file nobody maps and 3.4 GB/s on a mapped one, 8 MiB reads 8.6 GB/s.  So
    // windows less than kGap apart (the same role of nearby experts) are merged into requests of up to kMerge bytes.
    constexpr uint64_t kSector = 4096, kGap = 1ull << 20, kMerge = 32ull << 20;
    struct Window { int file; uint64_t a0, size, skip, n, at; uint8_t* dst; size_t req; uint64_t in_req; };
    struct Req { HANDLE h; uint64_t a0, size, pos; };
    // this thread's aligned buffer and events, kept for its next batch
    struct Scratch {
        uint8_t* buf = nullptr;
        size_t cap = 0;
        std::vector<HANDLE> ev;
        std::vector<OVERLAPPED> ov;
        std::vector<Window> win;
        std::vector<size_t> order;
        std::vector<Req> req;
        std::vector<DWORD> got;
        ~Scratch() {
            if (buf != nullptr) VirtualFree(buf, 0, MEM_RELEASE);
            for (HANDLE e : ev) CloseHandle(e);
        }
    };
    thread_local Scratch sc;
    sc.win.clear();
    for (size_t k = 0; k < n; ++k) {
        const Fill& f = fills[k];
        if (f.dst == nullptr || f.layer < 0 || f.e < 0 || f.layer >= n_layers_ || f.e >= n_expert_) return false;
        if (role_ptr_.empty()) {   // experts.bin: the blob is one contiguous range
            const uint64_t per = layer_blob_bytes_[(size_t) f.layer];
            const uint64_t off = layer_offsets_[(size_t) f.layer] + (uint64_t) f.e * per;
            const uint64_t a0 = off / kSector * kSector, a1 = (off + per + kSector - 1) / kSector * kSector;
            sc.win.push_back({0, a0, a1 - a0, off - a0, per, 0, f.dst, 0, 0});
            continue;
        }
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * f.layer + r);
            const uint64_t per = role_bytes_[i];
            const Map& m = maps_[(size_t) role_file_[i]];
            const uint64_t off = (uint64_t) (role_ptr_[i] - m.base) + (uint64_t) f.e * per;
            const uint64_t a0 = off / kSector * kSector, a1 = (off + per + kSector - 1) / kSector * kSector;
            sc.win.push_back({role_file_[i], a0, a1 - a0, off - a0, per, at, f.dst, 0, 0});
            at += per;
        }
    }
    sc.order.resize(sc.win.size());
    for (size_t w = 0; w < sc.win.size(); ++w) sc.order[w] = w;
    std::sort(sc.order.begin(), sc.order.end(), [&](size_t a, size_t b) {
        return sc.win[a].file != sc.win[b].file ? sc.win[a].file < sc.win[b].file : sc.win[a].a0 < sc.win[b].a0;
    });
    sc.req.clear();
    uint64_t total = 0;
    int last_file = -1;
    for (size_t w : sc.order) {
        Window& x = sc.win[w];
        if (!sc.req.empty() && x.file == last_file) {
            Req& q = sc.req.back();
            const uint64_t end = q.a0 + q.size, xend = x.a0 + x.size;
            if (x.a0 <= end + kGap && std::max(end, xend) - q.a0 <= kMerge) {
                if (xend > end) {
                    total += xend - end;
                    q.size = xend - q.a0;
                }
                x.req = sc.req.size() - 1;
                x.in_req = x.a0 - q.a0;
                continue;
            }
        }
        sc.req.push_back({(HANDLE) direct_[(size_t) x.file], x.a0, x.size, total});
        total += x.size;
        last_file = x.file;
        x.req = sc.req.size() - 1;
        x.in_req = 0;
    }
    if (total > sc.cap) {
        if (sc.buf != nullptr) VirtualFree(sc.buf, 0, MEM_RELEASE);
        sc.cap = (size_t) ((total + (1u << 20) - 1) >> 20 << 20);
        sc.buf = (uint8_t*) VirtualAlloc(nullptr, sc.cap, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (sc.buf == nullptr) { sc.cap = 0; return false; }
    }
    while (sc.ev.size() < sc.req.size()) {
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (e == nullptr) return false;
        sc.ev.push_back(e);
    }
    sc.ov.assign(sc.req.size(), OVERLAPPED{});
    // every request in flight before the first wait: the drive sees the whole batch as one queue
    size_t issued = 0;
    bool ok = true;
    for (size_t q = 0; q < sc.req.size(); ++q) {
        const Req& r = sc.req[q];
        OVERLAPPED& o = sc.ov[q];
        o.Offset = (DWORD) r.a0;
        o.OffsetHigh = (DWORD) (r.a0 >> 32);
        o.hEvent = sc.ev[q];
        if (!ReadFile(r.h, sc.buf + r.pos, (DWORD) r.size, nullptr, &o) && GetLastError() != ERROR_IO_PENDING) {
            ok = false;
            break;
        }
        ++issued;
    }
    std::vector<DWORD>& got = sc.got;
    got.assign(sc.req.size(), 0);
    for (size_t q = 0; q < issued; ++q)
        if (!GetOverlappedResult(sc.req[q].h, &sc.ov[q], &got[q], TRUE)) ok = false;
    if (!ok || issued < sc.req.size()) return false;
    for (const Window& x : sc.win) {
        // a request may run past the end of the file: only the role's own bytes have to arrive
        if ((uint64_t) got[x.req] < x.in_req + x.skip + x.n) return false;
        std::memcpy(x.dst + x.at, sc.buf + sc.req[x.req].pos + x.in_req + x.skip, (size_t) x.n);
    }
    return true;
#else
    (void) fills; (void) n;
    return false;
#endif
}

void FileExpertSource::warm(int64_t layer, const int64_t* experts, int64_t n) {
    if (role_ptr_.empty() || n <= 0 || layer < 0 || layer >= n_layers_) return;
    uint32_t stamp;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stamp = (uint32_t) epoch_ + 1;
    }
#if defined(_WIN32)
    using Pvm = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const Pvm pvm = (Pvm) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
#endif
    for (int64_t j = 0; j < n; ++j) {
        const int64_t e = experts[j];
        if (e < 0 || e >= n_expert_) continue;
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
        if (complement_ready_ && index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement)
            continue;                                                    // in the RAM copy
        if (warm_stamp_) warm_stamp_[index].store(stamp, std::memory_order_relaxed);
        warm_count_.fetch_add(1, std::memory_order_relaxed);
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint8_t* p = role_ptr_[i] + (size_t) ((uint64_t) e * role_bytes_[i]);
#if defined(_WIN32)
            ranges.push_back({(PVOID) p, (SIZE_T) role_bytes_[i]});
#else
            const uintptr_t pg = 4096, a = (uintptr_t) p & ~(pg - 1);
            (void) madvise((void*) a, (size_t) ((uintptr_t) p + role_bytes_[i] - a), MADV_WILLNEED);
#endif
        }
    }
#if defined(_WIN32)
    if (pvm != nullptr && !ranges.empty()) (void) pvm(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
#endif
}

RouterLookahead::~RouterLookahead() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool RouterLookahead::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k,
                            ExpertSource* src, std::string& err) {
    if (src == nullptr || !src->warms()) { err = "RouterLookahead: the expert source does not warm"; return false; }
    if (n_embd % 8 != 0) { err = "RouterLookahead: n_embd is not a multiple of 8"; return false; }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "RouterLookahead: a router of another shape"; return false; }
    routers_ = std::move(routers);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_ = k < 1 ? 1 : k > (int) n_expert ? (int) n_expert : k;
    src_ = src;
    x_.assign((size_t) (8 * n_embd), 0.f);
    thread_ = std::thread([this] { run(); });
    return true;
}

void RouterLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res) {
    if (layer + 1 >= (int64_t) routers_.size() || n_tok <= 0 || x == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (busy_ || pending_) { skipped_.fetch_add(1, std::memory_order_relaxed); return; }
        n_tok_ = std::min<int64_t>(n_tok, 8);
        std::memcpy(x_.data(), x, (size_t) (n_tok_ * n_embd_) * sizeof(float));
        layer_ = layer + 1;
        host_res_ = host_res;
        pending_ = true;
    }
    cv_.notify_one();
}

void RouterLookahead::run() {
    std::vector<float> logits((size_t) (8 * n_expert_));
    std::vector<int32_t> order((size_t) n_expert_);
    std::vector<int64_t> want;
    for (;;) {
        int64_t layer, nt;
        const int32_t* host_res;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || pending_; });
            if (quit_) return;
            pending_ = false;
            busy_ = true;
            layer = layer_;
            nt = n_tok_;
            host_res = host_res_;
        }
        const auto t0 = std::chrono::steady_clock::now();
        want.clear();
        // The router dot is AVX2 (kq_avx2.cpp); a CPU without AVX2 (the experimental older-CPU builds) takes the AVX1
        // one (kq_avx1.cpp) or, without AVX, the same sums in plain C++.  From the Strata_Dirigo fork (rwkeyes): an
        // AVX-only Xeon E5-2687W died here (vpmovzxwd) on its first request.  An estimate only (which experts to
        // prefetch); the order of the additions differs between the three, the output does not depend on it.
        if (strata::kernels::cpu::cpu_avx2_ok()) {
            strata::kernels::cpu::bf16_rows_dot_multi(routers_[(size_t) layer].data(), (int) n_expert_, (int) n_embd_,
                                                      x_.data(), (int) nt, logits.data());
        } else if (strata::kernels::cpu::cpu_avx1_ok()) {
            strata::kernels::cpu::bf16_rows_dot_multi_avx1(routers_[(size_t) layer].data(), (int) n_expert_,
                                                           (int) n_embd_, x_.data(), (int) nt, logits.data());
        } else {
            // a mul and an add, not std::fma: without an FMA instruction that is a libm call per element (74x
            // slower on a Xeon E5-2665, measured by the fork)
            const uint16_t* rw = routers_[(size_t) layer].data();
            for (int64_t r = 0; r < n_expert_; ++r) {
                const uint16_t* wr = rw + (size_t) r * (size_t) n_embd_;
                for (int64_t t = 0; t < nt; ++t) {
                    const float* xr = x_.data() + (size_t) t * (size_t) n_embd_;
                    float acc = 0.0f;
                    for (int64_t c = 0; c < n_embd_; ++c) {
                        const uint32_t bits = (uint32_t) wr[c] << 16;
                        float wf;
                        std::memcpy(&wf, &bits, sizeof wf);
                        acc += wf * xr[c];
                    }
                    logits[(size_t) t * (size_t) n_expert_ + (size_t) r] = acc;
                }
            }
        }
        for (int64_t t = 0; t < nt; ++t) {
            const float* lt = logits.data() + (size_t) (t * n_expert_);
            for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;
            std::partial_sort(order.begin(), order.begin() + k_, order.end(),
                              [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });
            for (int j = 0; j < k_; ++j) {
                const int64_t e = order[(size_t) j];
                if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;   // on the GPU
                if (std::find(want.begin(), want.end(), e) == want.end()) want.push_back(e);
            }
        }
        src_->warm(layer, want.data(), (int64_t) want.size());
        predicted_.fetch_add((int64_t) want.size(), std::memory_order_relaxed);
        busy_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                           std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
        }
    }
}

void FileExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    (void) ids;
    (void) k;
    if (!staged()) return;
    std::lock_guard<std::mutex> lk(stage_mu_);
    if (layer != last_layer_) {
        ++epoch_;
        last_layer_ = layer;
    }
}

bool FileExpertSource::transient(int64_t layer, int64_t expert) const {
    if (!staged() || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (complement_ready_ && index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement)
        return false;
    return override_.empty() || override_[index] == nullptr;
}

bool FileExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (base_ == nullptr || dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_)
        return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint64_t bytes = layer_blob_bytes_[(size_t) layer];
    if (complement_ready_) {
        const uint8_t* held =
            detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, nullptr);
        if (held == nullptr && !override_.empty()) held = override_[index];
        if (held != nullptr) {
            std::memcpy(dst, held, (size_t) bytes);
            return true;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!copy_from_files(layer, expert, dst)) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (!role_ptr_.empty()) return nullptr;   // the GGUF in place: no contiguous blob in any file
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) layer;
    if (i >= layer_offsets_.size() || i >= layer_blob_bytes_.size()) return nullptr;
    const uint64_t blob_bytes = layer_blob_bytes_[i];
    if (blob_bytes == 0 || (uint64_t) expert > std::numeric_limits<uint64_t>::max() / blob_bytes) return nullptr;
    const uint64_t expert_offset = (uint64_t) expert * blob_bytes;
    const uint64_t layer_offset = layer_offsets_[i];
    if (layer_offset > mapped_bytes_ || expert_offset > mapped_bytes_ - layer_offset) return nullptr;
    const uint64_t offset = layer_offset + expert_offset;
    if (blob_bytes > mapped_bytes_ - offset) return nullptr;
    return base_ + (size_t) offset;
}

bool FileExpertSource::pin_cache_complement(
    const ExpertCache& cache, std::string& err, bool pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs, int64_t lend_from_slot,
    uint64_t headroom_bytes, uint64_t budget_bytes, const std::vector<std::pair<int32_t, int32_t>>* rank) {
    err.clear();
    if (base_ == nullptr) { err = "FileExpertSource: open the mapped experts before pinning a complement"; return false; }
    if (complement_ready_) { err = "FileExpertSource: the cache complement is already pinned"; return false; }
    if (!cache.valid()) { err = "FileExpertSource: the GPU expert cache is not open"; return false; }
    if (cache.fills() != cache.resident()) {
        err = "FileExpertSource: the GPU expert cache is not fully filled";
        return false;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        err = std::string("FileExpertSource: GPU expert cache is not ready: ") + cudaGetErrorString(sync);
        (void) cudaGetLastError();
        return false;
    }

    // The GPU cache's experts, and the bytes each slot's expert takes here (for the lend region below).
    const int64_t n_slots = cache.slots();
    std::vector<std::pair<int32_t, int32_t>> primary_gpu_pairs;
    std::vector<int32_t> pair_slot;
    std::vector<uint64_t> slot_bytes((size_t) std::max<int64_t>(n_slots, 0), 0);
    primary_gpu_pairs.reserve((size_t) cache.resident());
    pair_slot.reserve((size_t) cache.resident());
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const int32_t slot = cache.slot_of(layer, expert);
            if (slot == kNotResident) continue;
            primary_gpu_pairs.emplace_back((int32_t) layer, (int32_t) expert);
            pair_slot.push_back(slot);
            if (slot >= 0 && slot < n_slots) slot_bytes[(size_t) slot] = layer_blob_bytes_[(size_t) layer];
        }
    }
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, primary_gpu_pairs,
                                            additional_gpu_pairs, offsets, bytes, err)) return false;
#if defined(_WIN32)
    // #467: the GPU cache's pre-fill touched its experts through the mapping (~19 GiB on a 24 GB card), and Windows
    // counts those file pages in this process's working set, not as available: a 32 GB PC read 0.44 GiB here
    // (20.7 GiB before the start).  Trimmed, they move to the standby list (still cached, counted as available).
    // Resident mode only: nothing else calls this function.  Locked/pinned pages stay; the rest fault back softly.
    {
        uint64_t before = 0, after = 0;
        const bool read_before = available_memory_bytes(before);
        (void) SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T) -1, (SIZE_T) -1);
        if (read_before && available_memory_bytes(after))
            std::fprintf(stderr, "FileExpertSource: available RAM %.2f GiB, %.2f GiB after the mapped experts left the "
                                 "process working set (#467)\n",
                         (double) before / 1073741824.0, (double) after / 1073741824.0);
    }
#endif
    const bool what_fits = budget_bytes == kResidentWhatFits;   // #467: the soft mode's second try
    uint64_t budget_physical = 0;   // #403: the RAM reading a budget was sized from (0: no budget)
    if (budget_bytes > 0) {
        // CS-T: a RAM budget.  The complement's experts in `rank` order (the expert profile, hottest first) while
        // they fit, the rest left on the mapped files; clamped to what the RAM has room for.
        uint64_t physical = 0;
        if (!available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for --resident-budget-gib";
            return false;
        }
        budget_physical = physical;
        const uint64_t room = physical > headroom_bytes ? physical - headroom_bytes : 0;
        if (budget_bytes > room) {
            // #403: 256 MiB under the room, so the engine's own allocations after this reading still leave the
            // headroom (a budget clamped to exactly the room failed the safety check below on a reading a few MB
            // lower).  An unclamped budget is unchanged.
            const uint64_t margin = 256ull << 20;
            const uint64_t clamped = room > margin ? room - margin : 0;
            if (what_fits)
                std::fprintf(stderr, "FileExpertSource: RAM room for the complement: %.2f GiB (%.2f GiB available "
                                     "minus %.0f GiB headroom and a 0.25 GiB margin)\n",
                             (double) clamped / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0);
            else
                std::fprintf(stderr, "FileExpertSource: --resident-budget-gib %.2f is more than the RAM has room for "
                                     "(%.2f GiB available minus %.0f GiB headroom and a 0.25 GiB margin): %.2f GiB\n",
                             (double) budget_bytes / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0, (double) clamped / 1073741824.0);
            budget_bytes = clamped;
        }
        std::vector<uint64_t> ranked(offsets.size(), kNoComplement);
        uint64_t at = 0;
        int64_t held = 0;
        if (rank != nullptr)
            for (const auto& pr : *rank) {
                if (pr.first < 0 || pr.second < 0 || pr.first >= n_layers_ || pr.second >= n_expert_) continue;
                const size_t i = (size_t) pr.first * (size_t) n_expert_ + (size_t) pr.second;
                if (offsets[i] == kNoComplement || ranked[i] != kNoComplement) continue;   // on a GPU, or twice
                const uint64_t b = layer_blob_bytes_[(size_t) pr.first];
                if (b > budget_bytes - at) continue;
                ranked[i] = at;
                at += b;
                ++held;
            }
        if (what_fits && held == 0) {   // #467: nothing to keep - the caller's plain mmap fallback, not an empty copy
            err = "FileExpertSource: the RAM has no room for any expert of the complement";
            return false;
        }
        std::fprintf(stderr, "FileExpertSource: RAM budget %.2f GiB: %lld of the %.2f GiB of experts the GPU cache does "
                             "not hold, by profile rank; the rest are read from the files\n",
                     (double) budget_bytes / 1073741824.0, (long long) held, (double) bytes / 1073741824.0);
        offsets.swap(ranked);
        bytes = at;
        lend_from_slot = -1;
    }

    const bool lend = lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.empty();
    uint64_t budget = std::numeric_limits<uint64_t>::max();
    if (bytes > 0 || lend) {
        // #403: with a budget, the reading it was sized from - a second reading a few MB lower (the engine's own
        // allocations, the file cache) failed a budget the first one had clamped.  (A budget turns `lend` off.)
        uint64_t physical = budget_physical;
        if (physical == 0 && !available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for the resident-memory safety check";
            return false;
        }
        budget = physical > headroom_bytes ? physical - headroom_bytes : 0;
        if (bytes > budget) {
            char message[320];
            std::snprintf(message, sizeof message,
                          "FileExpertSource: resident complement %.2f GiB exceeds available RAM (%.2f GiB) minus the "
                          "%.0f GiB safety headroom",
                          (double) bytes / 1073741824.0, (double) physical / 1073741824.0,
                          (double) headroom_bytes / 1073741824.0);
            err = message;
            return false;
        }
    }
    // The prompt path's lend region: its slots' experts are streamed from here during a prompt and copied back into
    // their slots after it, so the ones that fit are kept here too (from the last slot down: a short prompt lends
    // only the last few).  The rest keep the mapped-file fallback.
    int64_t keep_from = n_slots;
    if (lend) {
        keep_from = detail::choose_resident_keep_from(slot_bytes, bytes, budget, lend_from_slot);
        if (keep_from < 0) keep_from = n_slots;
        if (keep_from < n_slots) {
            std::vector<std::pair<int32_t, int32_t>> core;
            core.reserve(primary_gpu_pairs.size());
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i)
                if (pair_slot[i] < keep_from) core.push_back(primary_gpu_pairs[i]);
            if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, core,
                                                    additional_gpu_pairs, offsets, bytes, err)) return false;
        }
    }

    void* arena = nullptr;
    const uint8_t* host = nullptr;
    const uint8_t* device = nullptr;
    bool pinned_ok = false;
    uint64_t locked = 0;
    uint64_t partial_pin = 0;   ///< CS-T: a registered prefix of a locked arena
    uint64_t lock_off = 0;      ///< where the working-set lock starts (after the registered prefix)
    std::string note;
    auto release = [&]() {
        if (arena == nullptr) return;
        if (pinned_ok) (void) cudaFreeHost(arena);
        else {
            if (partial_pin > 0) (void) cudaHostUnregister(arena);
            if (locked > 0) strata::platform::unlock_resident((uint8_t*) arena + lock_off, locked);
            std::free(arena);
        }
        arena = nullptr;
    };
    if (bytes > 0) {
        std::fprintf(stderr, "FileExpertSource: allocating %.2f GiB %s cache complement\n",
                     (double) bytes / 1073741824.0, pin ? "page-locked" : "pageable resident");
        std::fflush(stderr);
        if (pin) {
            const cudaError_t allocated = cudaHostAlloc(&arena, (size_t) bytes,
                                                         cudaHostAllocMapped | cudaHostAllocPortable);
            if (allocated == cudaSuccess) {
                void* alias = nullptr;
                const cudaError_t aliased = cudaHostGetDevicePointer(&alias, arena, 0);
                if (aliased == cudaSuccess && alias != nullptr) {
                    device = (const uint8_t*) alias;
                    pinned_ok = true;
                    note = "page-locked and mapped";
                } else {
                    note = std::string("no device alias (") + cudaGetErrorString(aliased) + ")";
                    (void) cudaGetLastError();
                    (void) cudaFreeHost(arena);
                    arena = nullptr;
                }
            } else {
                // Refused (the driver's page-locked limit): the same bytes in ordinary memory, locked in the working
                // set instead, as the arena does - resident either way, only copied by the CPU instead of by DMA.
                note = std::string("page-locking refused (") + cudaGetErrorString(allocated) + ")";
                (void) cudaGetLastError();
                arena = nullptr;
            }
        }
        if (arena == nullptr) {
            arena = std::malloc((size_t) bytes);
            if (arena == nullptr) {
                err = "FileExpertSource: pageable resident complement allocation failed";
                return false;
            }
            if (pin) {
                // CS-T, a RAM budget: its bytes are in profile order, hottest first, so the driver is asked to
                // register the largest prefix it takes (from the cap down in 2 GiB steps).  Those experts can be
                // read by the GPU over PCIe (--pcie-frac) and copied by DMA; only the rest is locked in the working
                // set (the registered prefix is page-locked by the driver already - locking it twice made the next
                // device allocation fail).
                // opt-in (STRATA_PARTIAL_PIN=1): on the RTX 5070 PC the GPU's PCIe share of the misses measured no
                // faster than the CPU computing them (7.30 / 7.44 tok/s with 24 / 16 GiB registered against 7.05-7.74
                // unpinned at a 40 GiB budget), and registering adds startup time and driver memory pressure
                static const bool partial_on = [] {
                    const char* v = std::getenv("STRATA_PARTIAL_PIN");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                // at most STRATA_PARTIAL_PIN_GIB (default 24): registering 30 GiB of a 40 GiB arena left the driver
                // unable to page-lock the prompt path's small buffers afterwards (RTX 5070, WDDM)
                static const uint64_t pin_cap = [] {
                    const char* v = std::getenv("STRATA_PARTIAL_PIN_GIB");
                    return (uint64_t) ((v != nullptr && std::atof(v) > 0 ? std::atof(v) : 24.0) * 1073741824.0);
                }();
                if (budget_bytes > 0 && partial_on) {
                    const uint64_t step = 2ull << 30;
                    for (uint64_t want = std::min(bytes, pin_cap); want >= step; want = want > step ? want - step : 0) {
                        // cut at an expert boundary: a blob that started inside the registered range and ran past it
                        // would be taken as page-locked by a cudaMemcpyAsync and refused ("adaptive refill failed")
                        uint64_t w = want;
                        for (size_t i = 0; i < offsets.size(); ++i) {
                            if (offsets[i] == kNoComplement) continue;
                            const uint64_t b = layer_blob_bytes_[i / (size_t) n_expert_];
                            if (offsets[i] < want && offsets[i] + b > want) { w = offsets[i]; break; }
                        }
                        if (w == 0) break;
                        if (cudaHostRegister(arena, (size_t) w, cudaHostRegisterMapped | cudaHostRegisterPortable) ==
                            cudaSuccess) {
                            void* alias = nullptr;
                            if (cudaHostGetDevicePointer(&alias, arena, 0) == cudaSuccess && alias != nullptr) {
                                device = (const uint8_t*) alias;
                                partial_pin = w;
                            } else {
                                (void) cudaGetLastError();
                                (void) cudaHostUnregister(arena);
                            }
                            break;
                        }
                        (void) cudaGetLastError();
                        if (want <= step) break;
                    }
                    char msg[160];
                    std::snprintf(msg, sizeof msg, "%.2f GiB of it registered for the GPU (the hottest)",
                                  (double) partial_pin / 1073741824.0);
                    note += std::string("; ") + msg;
                }
                lock_off = partial_pin;
                const strata::platform::LockResult lr =
                    strata::platform::lock_resident((uint8_t*) arena + lock_off, bytes - lock_off);
                locked = lr.locked_bytes;
                note += (note.empty() ? "" : "; ") + lr.note;
            }
        }
        host = (const uint8_t*) arena;
    }

    // Copied layer by layer on a few threads: the page faults of the mapped file are the cost, and they overlap.
#if !defined(_WIN32)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        err = "FileExpertSource: cannot determine page size for mapped-page release";
        release();
        return false;
    }
#endif
    std::atomic<int64_t> next_layer{0}, layers_done{0};
    std::atomic<uint64_t> copied{0};
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    std::string fail_msg;
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> lock(fail_mu);
        if (fail_msg.empty()) fail_msg = m;
        failed.store(true);
    };
    auto worker = [&]() {
        for (;;) {
            const int64_t layer = next_layer.fetch_add(1);
            if (layer >= n_layers_ || failed.load()) return;
            const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
            std::vector<Fill> batch;   // #286, unbuffered: 32 blobs' reads in flight at once, merged where near
            auto flush = [&]() {
                if (batch.empty()) return true;
                bool ok = read_direct(batch.data(), batch.size());
                for (size_t i = 0; !ok && i < batch.size(); ++i)
                    if (!copy_from_files(batch[i].layer, batch[i].e, batch[i].dst)) return false;
                copied.fetch_add(blob_bytes * (uint64_t) batch.size());
                batch.clear();
                return true;
            };
            for (int64_t expert = 0; expert < n_expert_; ++expert) {
                const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
                const uint64_t offset = offsets[index];
                if (offset == kNoComplement) continue;
                if (offset > bytes || blob_bytes > bytes - offset) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                uint8_t* dst = (uint8_t*) host + (size_t) offset;
                if (!direct_.empty()) {
                    batch.push_back({0, layer, expert, dst});
                    if (batch.size() == 32 && !flush()) {
                        fail("FileExpertSource: an expert could not be read while building the cache complement");
                        return;
                    }
                    continue;
                }
                if (!copy_from_files(layer, expert, dst)) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                copied.fetch_add(blob_bytes);
            }
            if (!flush()) {
                fail("FileExpertSource: an expert could not be read while building the cache complement");
                return;
            }
#if !defined(_WIN32)
            if (role_ptr_.empty()) {   // experts.bin; the GGUF in place leaves its pages to the OS
            const uint64_t layer_offset = layer_offsets_[(size_t) layer];
            const uint64_t layer_bytes = blob_bytes * (uint64_t) n_expert_;
            const uint64_t layer_end = layer_offset + layer_bytes;
            const uint64_t page = (uint64_t) page_size;
            const uint64_t advice_start = layer_offset - layer_offset % page;
            const uint64_t end_remainder = layer_end % page;
            const uint64_t extra = end_remainder == 0 ? 0 : page - end_remainder;
            const uint64_t advice_end = extra > mapped_bytes_ - layer_end ? mapped_bytes_ : layer_end + extra;
            if (advice_end > advice_start &&
                madvise((void*) (base_ + (size_t) advice_start), (size_t) (advice_end - advice_start), MADV_DONTNEED) != 0) {
                fail("FileExpertSource: madvise could not release mapped expert layer " + std::to_string(layer));
                return;
            }
            if (posix_fadvise(fd_, (off_t) layer_offset, (off_t) layer_bytes, POSIX_FADV_DONTNEED) != 0) {
                fail("FileExpertSource: posix_fadvise could not release expert layer " + std::to_string(layer));
                return;
            }
            }
#endif
            const int64_t done = layers_done.fetch_add(1) + 1;
            if (done % 8 == 0 || done == n_layers_)
                std::fprintf(stderr, "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB)\n",
                             (long long) done, (long long) n_layers_, (double) copied.load() / 1073741824.0);
        }
    };
    {
        const int threads = (int) std::max<int64_t>(1, std::min<int64_t>(6, n_layers_));
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    std::fflush(stderr);
    if (failed.load()) {
        err = fail_msg;
        release();
        return false;
    }
#if defined(_WIN32)
    // The mapped pages this process touched (the GPU cache's fill and this copy) leave its working set for the
    // standby list: VirtualUnlock on pages that are not locked does exactly that (it then reports ERROR_NOT_LOCKED).
    if (maps_.empty()) (void) VirtualUnlock((LPVOID) base_, (SIZE_T) mapped_bytes_);
    for (const Map& m : maps_) (void) VirtualUnlock((LPVOID) m.base, (SIZE_T) m.bytes);
#endif

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    complement_pinned_ = (pinned_ok || partial_pin > 0) && bytes > 0;
    complement_pin_limit_ = pinned_ok ? bytes : partial_pin;
    complement_partial_ = !pinned_ok && partial_pin > 0;
    complement_locked_ = locked;
    complement_lock_off_ = lock_off;
    complement_lent_slots_ = lend ? n_slots - keep_from : 0;
    complement_ready_ = true;
    std::fprintf(stderr, "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB%s%s\n",
                 complement_pinned_ ? "mapped pinned" : pin ? "locked resident" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0,
                 note.empty() ? "" : "; ", note.c_str());
    if (lend)
        std::fprintf(stderr, "FileExpertSource: %lld of the prompt path's %lld lendable slots keep their experts in RAM "
                             "too%s\n", (long long) complement_lent_slots_, (long long) (n_slots - lend_from_slot),
                     complement_lent_slots_ < n_slots - lend_from_slot
                         ? " (the others are read from the file when lent: not enough RAM for them)" : "");
    if (!additional_gpu_pairs.empty()) {
        std::fprintf(stderr, "FileExpertSource: %zu verified additional-GPU experts remain on the mmap fallback\n",
                     additional_gpu_pairs.size());
    }
    std::fflush(stderr);
    return true;
}

bool FileExpertSource::has_resident(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement;
}

bool FileExpertSource::reserve_exchanges(int64_t n, std::string& err) {
    err.clear();
    if (n <= xstage_cap_) return true;
    if (!staged_.empty()) { err = "FileExpertSource: exchange buffers are in use"; return false; }
    uint64_t blob = 0;
    for (const uint64_t b : layer_blob_bytes_) blob = std::max(blob, b);
    if (blob == 0 || n <= 0) { err = "FileExpertSource: no expert geometry for the exchange buffers"; return false; }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
        xstage_ = nullptr;
        xstage_cap_ = 0;
    }
    const size_t total = (size_t) n * (size_t) blob;
    void* p = nullptr;
    if (cudaHostAlloc(&p, total, cudaHostAllocDefault) == cudaSuccess && p != nullptr) {
        xstage_pinned_ = true;
    } else {
        (void) cudaGetLastError();
        p = std::malloc(total);
        xstage_pinned_ = false;
        if (p == nullptr) { err = "FileExpertSource: cannot allocate the exchange buffers"; return false; }
    }
    xstage_ = (uint8_t*) p;
    xstage_cap_ = n;
    xstage_blob_ = blob;
    return true;
}

uint8_t* FileExpertSource::exchange_buffer(int64_t q) const {
    if (xstage_ == nullptr || q < 0 || q >= xstage_cap_) return nullptr;
    return xstage_ + (size_t) q * (size_t) xstage_blob_;
}

bool FileExpertSource::stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q) {
    if (!has_resident(layer, in) || has_resident(layer, out) || exchange_buffer(q) == nullptr) return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (override_.empty()) override_.assign((size_t) blobs_, nullptr);
    if (override_[i_out] != nullptr) return false;
    for (const Exchange& x : staged_)
        if (x.in == i_in || x.q == q) return false;
    override_[i_out] = exchange_buffer(q);
    staged_.push_back({i_in, i_out, q, layer_blob_bytes_[(size_t) layer]});
    return true;
}

int64_t FileExpertSource::commit_exchanges() {
    int64_t n = 0;
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_[x.out];
        const uint64_t at = complement_offsets_[x.in];
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr) {
            std::memcpy((uint8_t*) complement_host_ + (size_t) at, src, (size_t) x.bytes);
            if (detail::exchange_cache_complement(complement_offsets_, x.in, x.out)) ++n;
        }
        override_[x.out] = nullptr;
    }
    staged_.clear();
    exchanges_ += n;
    return n;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint8_t* result = nullptr;
    bool from_files = true;
    if (complement_ready_) {
        result = detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, nullptr);
        if (result != nullptr) {
            ram_reads_.fetch_add(1, std::memory_order_relaxed);
            from_files = false;
        } else if (!override_.empty() && override_[index] != nullptr) {
            result = override_[index];
            from_files = false;
        }
    }
    if (from_files) {
        if (staged()) {
            result = staged_blob(layer, expert);          // counts its bytes
        } else {
            result = mapped_blob(layer, expert);
            if (result != nullptr)
                file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        }
        if (complement_ready_ && result != nullptr) file_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    if (result != nullptr) ++reads_;
    return result;
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || !complement_pinned_ || complement_host_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (index >= complement_offsets_.size() || complement_offsets_[index] == kNoComplement) return false;
    // a partial pin (CS-T): only the registered prefix
    return !complement_partial_ ||
           complement_offsets_[index] + layer_blob_bytes_[(size_t) layer] <= complement_pin_limit_;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || complement_device_ == nullptr) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return complement_device_ + (size_t) complement_offsets_[index];
}

bool FileExpertSource::pcie_layer(int64_t layer) const {
    if (complement_ready_ && complement_pinned_ && complement_device_ != nullptr)
        return layer >= 0 && layer < n_layers_;
    return device_alias(layer, 0) != nullptr;
}

// ================================ THE ADAPTER =========================
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);
    // R7: this path has nothing to interleave - one token, one batch - so it takes the old single-phase shape.
    d.src->wait_layer();

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

namespace {
// the verify window's per-entry tables in `expert_pool_dispatch_multi` (`kind`, `distinct`, `first_of`)
// are fixed arrays of this many entries: MAXT tokens of the model's 10 routed experts must fit, and a larger k is
// refused at run time rather than written past them.
constexpr int64_t kMaxWindowEntries = 128;
static_assert(strata::kernels::cpu::MAXT * 10 <= kMaxWindowEntries, "a verify window's entries overflow the tables");
}  // namespace

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if (d.lookahead != nullptr) d.lookahead->submit(d.layers, x_f, n_tok, d.host_res);   // CS-T: warm layer + 1
    if (k < 1 || n_tok * k > kMaxWindowEntries) {
        d.failed = true;
        d.fail = "a verify window routes more entries than the expert pool's window tables hold";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[kMaxWindowEntries];       // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap) {
        int64_t distinct[kMaxWindowEntries], first_of[kMaxWindowEntries];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0 &&
                    !(d.peer != nullptr && d.peer->has(d.layers, e))) ++nmiss;
            }
        }
        const bool pcie_ok = d.pcie_num > 0 && d.src->pcie_layer(d.layers);
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else if (d.peer != nullptr && d.peer->has(d.layers, e)) {
                    kd = 2;                        // multi-GPU: the second GPU computes it
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src = d.src->pinned(d.layers, e) ? d.src->blob(d.layers, e) : nullptr;
                        if (src != nullptr) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P.publish) P.publish(P.ctx);
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0
                    : (e >= 0 && e < d.n_expert && d.peer != nullptr && d.peer->has(d.layers, e)) ? 2 : -1;
        }
    }
    if (d.peer != nullptr) {                 // multi-GPU: start the second GPU's share before the CPU's own work
        std::string perr;
        if (!d.peer->launch(d.layers, x_f, ids, n_tok, k, kind, perr, out)) {
            std::fprintf(stderr, "strata: %s (layer %lld)\n", perr.c_str(), (long long) d.layers);
            d.failed = true;
            d.fail = "the peer GPU's experts could not be launched";
            d.fail_layer = d.layers;
            return;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    // the GPU's copy_rows_from_mapped zeroes the rows it computed itself (dec_batch); STRATA_VERIFY_DEVICE_PLAN's
    // copy_or_zero_from_mapped copies every row, so there the host still zeroes them
    static const bool device_plan = [] { const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN"); return v != nullptr && std::atoi(v) != 0; }();
    const bool gpu_zeroes_hits = (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap && dec_batch && !device_plan);
    bool any_cpu = false;
    for (int64_t i = 0; i < n; ++i)
        if (kind[i] < 0) { any_cpu = true; break; }
    const auto c1 = std::chrono::steady_clock::now();
    if (any_cpu) {
        // #578 --remote-expert-opt: a token whose experts all run on a GPU (CUDA0 or a helper) needs no CPU activation
        const bool ep = d.remote_count > 0 && d.remote[0]->optimized_decode();
        for (int64_t t = 0; t < n_tok; ++t) {
            if (ep && std::all_of(kind + t * k, kind + (t + 1) * k, [](int32_t v) { return v >= 0; })) continue;
            if (native && strata::kernels::cpu::q2_native_kernels(lay.fmt[(size_t) d.layers].gu_type))   // a native Q2_0 pack: the Q2_0 kernels' activations
                act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
            else if (native)
                native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
            else
                act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
        }
    }
    const auto c2 = std::chrono::steady_clock::now();
    if (any_cpu) {   // CS-T: the experts the CPU computes, fetched together (the GGUF in place reads them on several threads)
        static thread_local std::vector<int64_t> miss;
        miss.clear();
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (kind[i] < 0 && ids[i] >= 0 && ids[i] < d.n_expert &&
                std::find(miss.begin(), miss.end(), (int64_t) ids[i]) == miss.end())
                miss.push_back(ids[i]);
        d.src->prefetch(d.layers, miss.data(), (int64_t) miss.size());
    }
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote/peer result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                else ++d.offload_entries;                       // #588: PCIe or another GPU
                if (kind[i] == 2 && d.peer != nullptr) ++d.peer_entries;
                // multi-GPU: a direct peer launch is writing this row right now - zeroing it would race it;
                // dec_batch: copy_rows_from_mapped_kernel already zeroes kind 0/1 rows on the GPU
                if (!((kind[i] == 2 && d.peer != nullptr && d.peer->launched_direct()) ||
                      (gpu_zeroes_hits && (kind[i] == 0 || kind[i] == 1))))
                    std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    if (njobs > 0) {
        if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
        else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    if (d.peer != nullptr) {                 // multi-GPU: the second GPU's rows, into the same mapped rows
        std::string perr;
        if (!d.peer->finish(out, perr)) {
            std::fprintf(stderr, "strata: %s (layer %lld)\n", perr.c_str(), (long long) d.layers);
            d.failed = true;
            d.fail = "the peer GPU's experts failed";
            d.fail_layer = d.layers;
            return;
        }
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS =========================        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) cudaEventRecord((cudaEvent_t) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) =========================
// R13: O_DIRECT helpers.  A blob's file offset is 256-byte aligned but rarely 4 KiB, its length is not a
// 4 KiB multiple, and the ring destinations are plain heap vectors - none of which O_DIRECT accepts.  Every
// direct read therefore covers the blob with the enclosing 4 KiB-aligned span into a thread-local aligned
// bounce buffer, and the exact blob bytes are memcpy'd out of it.  One extra 2.18 MB copy (~0.1 ms) against
// the page-cache alloc/copy/DONTNEED cycle it removes.
namespace {

/// Full pread; a short read on a valid regular fd is EOF-level trouble, so it fails the read.
bool pread_full(int fd, void* dst, size_t n, off_t off) {
    uint8_t* p = (uint8_t*) dst;
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, p + done, n - done, off + (off_t) done);
        if (r <= 0) return false;
        done += (size_t) r;
    }
    return true;
}

void hash_u64(uint64_t& h, uint64_t v) {
    h = fnv1a64((const uint8_t*) &v, sizeof v, h);
}

void hash_text(uint64_t& h, const std::string& s) {
    h = fnv1a64((const uint8_t*) s.data(), (uint64_t) s.size(), h);
}

bool hash_small_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    hash_text(h, path.filename().string());
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        hash_u64(h, 0);
        return true;
    }
    hash_u64(h, 1);
    std::vector<uint8_t> buf(64u << 10);
    for (;;) {
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        const std::streamsize n = f.gcount();
        if (n > 0) h = fnv1a64(buf.data(), (uint64_t) n, h);
        if (f.eof()) break;
        if (!f) {
            err = "ArenaExpertSource: cannot hash pack metadata " + path.string();
            return false;
        }
    }
    return true;
}

/// One aligned bounce buffer per thread, grown to the largest blob (plus the 4 KiB head/tail over-read).
uint8_t* tls_bounce(size_t bytes) {
    struct Buf { void* p = nullptr; ~Buf() { std::free(p); } };
    thread_local Buf buf;
    thread_local size_t cap = 0;
    if (cap < bytes) {
        void* q = nullptr;
        if (posix_memalign(&q, 4096, bytes) != 0) return nullptr;
        std::free(buf.p);
        buf.p = q;
        cap = bytes;
    }
    return (uint8_t*) buf.p;
}

/// R20: what alignment does O_DIRECT *actually* require for this file?  The code assumed 4096.  Here the NVMe
/// reports logical_block_size 512 and btrfs accepts 512-aligned direct reads, so that assumption was 8x
/// stricter than necessary - and because a blob's stride is 2,176,000 = 2^10 x 2125 bytes, only 1 blob in 4
/// sits on a 4 KiB boundary.  The other 75% paid a full-size bounce + memcpy on EVERY miss.
///
/// Probed rather than taken from sysfs: a real O_DIRECT pread at an offset that is a multiple of 512 but NOT
/// of 1024 succeeds only if the whole stack (fs + block layer + device) truly permits 512.  That tests the
/// exact operation we are about to rely on, which a sysfs value or a statx claim does not.  Anything
/// unexpected keeps the old, always-safe 4096 behaviour, so this cannot break another filesystem.
uint64_t probe_direct_alignment(int fd, uint64_t file_bytes) {
    constexpr uint64_t kProbeLen = 512;
    uint64_t off = 512;                        // 512-aligned, deliberately NOT 1024-aligned
    if (file_bytes < off + kProbeLen) return 4096;
    void* buf = nullptr;
    if (posix_memalign(&buf, 4096, kProbeLen) != 0) return 4096;  // a 4096-aligned buffer is never the limiter
    const ssize_t r = ::pread(fd, buf, (size_t) kProbeLen, (off_t) off);
    std::free(buf);
    if (r != (ssize_t) kProbeLen) return 4096;  // EINVAL or anything else: stay on the strict, safe path
    return 512;
bool hash_sampled_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "ArenaExpertSource: cannot sample pack source " + path.string();
        return false;
    }
    const std::streamoff end = f.tellg();
    if (end < 0) {
        err = "ArenaExpertSource: cannot size pack source " + path.string();
        return false;
    }
    const uint64_t bytes = (uint64_t) end;
    hash_text(h, path.filename().string());
    hash_u64(h, bytes);
    constexpr uint64_t sample = 64u << 10;
    const uint64_t starts[3] = {0, bytes / 2, bytes > sample ? bytes - sample : 0};
    std::vector<uint8_t> buf((size_t) std::min<uint64_t>(sample, bytes));
    for (uint64_t off : starts) {
        if (buf.empty()) break;
        const uint64_t at = std::min<uint64_t>(off, bytes - (uint64_t) buf.size());
        f.clear();
        f.seekg((std::streamoff) at);
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        if ((size_t) f.gcount() != buf.size()) {
            err = "ArenaExpertSource: short read while hashing pack source " + path.string();
            return false;
        }
        hash_u64(h, at);
        h = fnv1a64(buf.data(), (uint64_t) buf.size(), h);
    }
    return true;
}

bool shared_arena_pack_hash(const std::string& pack_dir, const std::string& experts_path,
                            const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay,
                            uint64_t& out, std::string& err) {
    uint64_t h = 1469598103934665603ull;
    hash_text(h, "strata-shared-expert-arena-pack-v1");
    hash_u64(h, (uint64_t) lay.n_layers);
    hash_u64(h, (uint64_t) lay.n_expert);
    hash_u64(h, lay.total);
    hash_u64(h, lay.max_blob);
    hash_u64(h, lay.native ? 1 : 0);

    const std::filesystem::path pack(pack_dir);
    for (const char* name : {"manifest.json", "index.txt", "native_experts.txt"}) {
        if (!hash_small_file(pack / name, h, err)) return false;
    }

    if (std::filesystem::exists(experts_path)) {
        if (!hash_sampled_file(experts_path, h, err)) return false;
    } else if (!gguf.empty()) {
        // Native packs may read experts straight from one or more GGUF shards.  Sample every distinct source
        // file named by native_experts.txt; this keeps the fingerprint cheap while still tying it to the model
        // bytes rather than only to an equal-size layout.
        const std::filesystem::path first(gguf);
        std::vector<std::filesystem::path> sources{first};
        for (const std::string& name : lay.gguf_file) {
            if (name.empty()) continue;
            const std::filesystem::path p = first.parent_path() / name;
            if (std::find(sources.begin(), sources.end(), p) == sources.end()) sources.push_back(p);
        }
        for (const auto& p : sources) {
            if (!hash_sampled_file(p, h, err)) return false;
        }
    }

    out = h == 0 ? 1 : h;
    return true;
}

}  // namespace

// R13: read the blob's bytes through `direct_fd_` into `dst`.  Returns the blob length, or -1 on any
// failure (the caller falls back to the buffered pread, so a exotic filesystem cannot break the engine).
int64_t ArenaExpertSource::direct_read_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (!direct_ok_ || direct_fd_ < 0 || dst == nullptr) return -1;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return -1;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0) return -1;
    const uint64_t off = lay.blob_offset(layer, expert);
    if (off + len > file_map_bytes_) return -1;
    constexpr uint64_t A = 4096;
    const uint64_t off_al = off & ~(A - 1);
    const uint64_t span = ((off + len + A - 1) & ~(A - 1)) - off_al;
    uint8_t* bounce = tls_bounce((size_t) span);
    if (bounce == nullptr) return -1;
    if (!pread_full(direct_fd_, bounce, (size_t) span, (off_t) off_al)) return -1;
    std::memcpy(dst, bounce + (off - off_al), (size_t) len);
    return (int64_t) len;
}

// R13: read `len` bytes at file offset `off` (arbitrary alignment) through `direct_fd_` into `dst`.
//
// R20: the alignment used to be hard-coded 4096, which forced EVERY miss through a full-size bounce buffer
// plus a 2.18 MB memcpy: a blob's stride is 2,176,000 = 2^10 x 2125 bytes, so only 1 blob in 4 starts on a
// 4 KiB boundary.  The device's real requirement is its 512-byte logical block size (probed at open into
// `direct_align_`), which every blob offset (1024-aligned) and every blob length (512-aligned) already
// satisfies - so the read now lands directly in its destination and the copy is gone.  At ~260 misses/token
// that was ~1.1 GB/s of read+write DRAM traffic stolen from the CPU expert kernels, which are themselves
// bandwidth-bound, plus the CPU on 10-16 reader threads.
bool ArenaExpertSource::direct_read_span(uint64_t off, uint64_t len, uint8_t* dst) {
    if (!direct_ok_ || direct_fd_ < 0 || dst == nullptr || len == 0) return false;
    if (off + len > file_map_bytes_) return false;
    const uint64_t A = direct_align_;
    const bool dst_al = ((reinterpret_cast<uintptr_t>(dst) & (uintptr_t)(A - 1)) == 0);
    if (((off | len) & (A - 1)) == 0 && dst_al)
        return pread_full(direct_fd_, dst, (size_t) len, (off_t) off);   // the no-copy path
    // Something is genuinely unaligned for this device: cover the blob with the enclosing aligned span in the
    // bounce buffer, exactly as R13 did.  Correct on every filesystem, just slower.
    const uint64_t off_al = off & ~(A - 1);
    const uint64_t span = ((off + len + A - 1) & ~(A - 1)) - off_al;
    uint8_t* bounce = tls_bounce((size_t) span);
    if (bounce == nullptr) return false;
    if (!pread_full(direct_fd_, bounce, (size_t) span, (off_t) off_al)) return false;
    std::memcpy(dst, bounce + (off - off_al), (size_t) len);
    return true;
}

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
namespace {
/// The GGUF file that holds role `r` (0 gate, 1 up, 2 down) of layer `l`: a name beside the --native shard
/// (native_experts.txt v3 per layer, v4 per role), or the --native shard itself.
std::string expert_gguf_file(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, int64_t l, int r) {
    const size_t i = (size_t) (3 * l + r);
    if (lay.gguf_file.size() <= i || lay.gguf_file[i].empty()) return gguf;
    const size_t cut = gguf.find_last_of("/\\");
    return (cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1)) + lay.gguf_file[i];
}
}  // namespace

bool check_experts_gguf(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, std::string& err) {
    static const char* roles[3] = {"gate", "up", "down"};
    if (lay.gguf_off.size() != (size_t) (3 * lay.n_layers)) {
        err = "native_experts.txt has no GGUF offsets (a pack older than v2): repack it with tools/iq_pack.py";
        return false;
    }
    try {
        std::map<std::string, std::unique_ptr<strata::GgufFile>> files;
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const std::string path = expert_gguf_file(gguf, lay, l, r);
                auto& f = files[path];
                if (!f) f = std::make_unique<strata::GgufFile>(path);
                const std::string name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                const strata::TensorInfo* t = f->find(name);
                const uint64_t want_type = (uint64_t) (r < 2 ? fm.gu_type : fm.d_type);
                // GGUF order: dim 0 is the row (the input), dim 1 the rows, dim 2 the experts
                const uint64_t cols = (uint64_t) (r < 2 ? fm.n_embd : fm.n_ff);
                const uint64_t rows = (uint64_t) (r < 2 ? fm.n_ff : fm.n_embd);
                const uint64_t bytes = per[r] * (uint64_t) lay.n_expert;
                const uint64_t payload = f->file_size() - f->data_start();
                std::string why;
                if (t == nullptr) why = "is not in it";
                else if (t->type != want_type)
                    why = std::string("is ") + t->type_name() + ", the pack says type " + std::to_string(want_type);
                else if (t->shape.size() != 3 || t->shape[0] != cols || t->shape[1] != rows ||
                         t->shape[2] != (uint64_t) lay.n_expert)
                    why = "is not [" + std::to_string(cols) + ", " + std::to_string(rows) + ", " +
                          std::to_string(lay.n_expert) + "]";
                else if (strata::tensor_payload_bytes(*t) != bytes)
                    why = "is not " + std::to_string(per[r]) + " B per expert";
                else if (f->data_start() + t->offset != lay.gguf_off[(size_t) (3 * l + r)])
                    why = "starts at byte " + std::to_string(f->data_start() + t->offset) + ", the pack says " +
                          std::to_string(lay.gguf_off[(size_t) (3 * l + r)]);
                else if (t->offset > payload || bytes > payload - t->offset)
                    why = "runs past the end of the file (a truncated shard?)";
                if (!why.empty()) {
                    err = "the pack's native_experts.txt does not match the model: " + name + " in " + path + " " +
                          why + " - repack with tools/iq_pack.py from this model's shards";
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("native experts from the GGUF: ") + e.what();
        return false;
    }
}

// Plan v0.3 P6: the arena from the model's GGUF shards.  Each layer's gate, up and down tensors hold the 512
// experts one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.  Each role
// is read from its own file (native_experts.txt v4: a shard boundary can fall inside a layer; per role as in
// #255, gopinath87607).  The caller checks the spans first (check_experts_gguf).
// `unbuffered` (Windows, experts_unbuffered): each chunk's 4 KiB-aligned window is read with FILE_FLAG_NO_BUFFERING into
// an aligned buffer and scattered into the blobs - no copy through the file cache when the drive is read anyway.
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, bool unbuffered) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
#if defined(_WIN32)
    if (unbuffered) {
        uint64_t max_chunk = 0;
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const auto& fm = lay.fmt[(size_t) l];
            max_chunk = std::max<uint64_t>(max_chunk, std::max<uint64_t>(fm.up_off, lay.bytes[(size_t) l] - fm.down_off) * 16);
        }
        std::mutex err_mu;
        std::string err;
        auto worker = [&]() {
            constexpr uint64_t kSector = 4096;
            const uint64_t cap = (max_chunk + 2 * kSector + kSector - 1) / kSector * kSector;
            uint8_t* buf = (uint8_t*) VirtualAlloc(nullptr, (size_t) cap, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            HANDLE h = INVALID_HANDLE_VALUE;
            std::string open_name;
            auto fail = [&](const std::string& what) {
                std::lock_guard<std::mutex> g(err_mu);
                if (err.empty()) err = what;
                bad = true;
            };
            if (buf == nullptr) fail("cannot allocate a read buffer");
            for (;;) {
                const int64_t l = next.fetch_add(1);
                if (l >= lay.n_layers || bad) break;
                const auto& fm = lay.fmt[(size_t) l];
                const uint64_t blob = lay.bytes[(size_t) l];
                const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
                const uint64_t at[3] = {0, fm.up_off, fm.down_off};
                for (int r = 0; r < 3 && !bad; ++r) {
                    // the handle is kept while consecutive roles share a file (every layer of a v3 pack)
                    const std::string name = expert_gguf_file(gguf, lay, l, r);
                    if (name != open_name) {
                        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
                        const int wide = MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, nullptr, 0);
                        std::vector<wchar_t> w((size_t) std::max(wide, 1), L'\0');
                        if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, w.data(), wide);
                        h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                        if (h == INVALID_HANDLE_VALUE) {
                            fail("cannot open " + name + " (error " + std::to_string((unsigned long long) GetLastError()) + ")");
                            break;
                        }
                        open_name = name;
                    }
                    const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                    const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                    const uint64_t chunk = per[r] * 16;           // 16 experts per read
                    for (uint64_t done = 0; done < total; done += chunk) {
                        const uint64_t n = std::min<uint64_t>(chunk, total - done);
                        const uint64_t a0 = (src + done) / kSector * kSector;
                        const uint64_t a1 = (src + done + n + kSector - 1) / kSector * kSector;
                        OVERLAPPED ov{};
                        ov.Offset = (DWORD) a0;
                        ov.OffsetHigh = (DWORD) (a0 >> 32);
                        DWORD got = 0;
                        // the window may run past the end of the file: only the tensor's own bytes have to arrive
                        if (!ReadFile(h, buf, (DWORD) (a1 - a0), &got, &ov) || (uint64_t) got < src + done - a0 + n) {
                            fail("short unbuffered read of layer " + std::to_string(l) + " in " + name + " (error " +
                                 std::to_string((unsigned long long) GetLastError()) + ")");
                            break;
                        }
                        const uint8_t* q = buf + (src + done - a0);
                        for (uint64_t k = 0; k < n / per[r]; ++k) {
                            const uint64_t e = done / per[r] + k;
                            std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], q + k * per[r], (size_t) per[r]);
                        }
                    }
                }
            }
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
            if (buf != nullptr) VirtualFree(buf, 0, MEM_RELEASE);
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
        if (bad) {
            st.seconds = -1.0;
            st.ok = false;
            st.error = err.empty() ? "unreadable shard while reading the experts from the GGUF" : err;
            return st;
        }
        st.bytes = lay.total;
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return st;
    }
#else
    (void) unbuffered;
#endif
    auto worker = [&]() {
        // #230: `fread` on a `FILE*`, as load_experts_ranges (#89): MSVC's `std::ifstream::read` splits a request
        // into 4095-byte freads, which took this path to 0.02 GiB/s on a Windows install without experts.bin.
        // The guard closes the handle on every return.
        struct Closer {
            FILE* f = nullptr;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } file;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                // the handle is kept while consecutive roles share a file (every layer of a v3 pack)
                const std::string name = expert_gguf_file(gguf, lay, l, r);
                if (name != open_name) {
                    if (file.f != nullptr) std::fclose(file.f);
                    file.f = std::fopen(name.c_str(), "rb");
                    if (file.f == nullptr) { bad = true; return; }
                    open_name = name;
                }
                FILE* const f = file.f;
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    // 64-bit seek: a shard is tens of GB
                    if (STRATA_FSEEK64(f, src + done) != 0) { bad = true; return; }
                    if (std::fread(buf.data(), 1, (size_t) n, f) != (size_t) n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        st.ok = false;
        st.error = "short read or unreadable shard while reading the experts from the GGUF";
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err, uint64_t max_pinned_bytes,
                             const std::string& shared_arena_file) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (from_gguf) {
        // every (file, offset) of native_experts.txt must be the tensor it claims, of the pack's type and
        // dimensions and inside its file - before the allocation, so a pack of another model or a truncated
        // shard is a message rather than an arena of plausible wrong experts
        if (!check_experts_gguf(gguf_, lay, err)) { err = "ArenaExpertSource: " + err; return false; }
    } else {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want && got != want + (uint64_t) blob) {   // + one blob: STRATA_ARENA_MMAP's padded file
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // STRATA_ARENA_MMAP=1 (a small-RAM machine whose GPUs hold most experts): the arena is the pack's experts.bin
    // mapped READ-ONLY - not locked, not pinned.  The page cache keeps what the CPU pool and the prompt path
    // actually read (the experts no VRAM cache holds) and gives back the rest under memory pressure; a page
    // dropped is re-read from the file, so a result never depends on what is resident.  The GPUs then get no
    // mapped alias: run with --pcie-frac 0.  The first start writes experts.bin (arena layout, padded by one blob
    // so a whole-slot copy may start at any expert), later ones map it.
#if defined(_WIN32)
    static const bool arena_mmap = false;   // POSIX mmap/madvise: Linux only for now
#else
    static const bool arena_mmap = [] { const char* v = std::getenv("STRATA_ARENA_MMAP"); return v && v[0] == '1'; }();
    if (arena_mmap) {
        const uint64_t file_bytes = want + (uint64_t) blob;
        uint64_t have = 0;
        {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (f) have = (uint64_t) f.tellg();
        }
        if (have == file_bytes) {
            const int fd = ::open(path.c_str(), O_RDONLY);
            void* v = fd >= 0 ? mmap(nullptr, (size_t) file_bytes, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
            if (fd >= 0) ::close(fd);
            if (v == MAP_FAILED) { err = "ArenaExpertSource: mmap of " + path + " failed"; return false; }
            map_ = v;
            map_bytes_ = file_bytes;
            base_ = (const uint8_t*) v;
            pinned_bytes_ = 0;
            dev_slice_.clear();
            slice_bytes_ = 0;
            blobs_ = n_layers * n_expert;
            n_expert_ = n_expert;
            reads_ = 0;
            note_ = "mapped read-only from " + path + " (STRATA_ARENA_MMAP: not locked, not pinned)";
            gib_per_s_ = 0.0;
            load_seconds_ = load_read_s_ = load_copy_s_ = 0.0;
            return true;
        }
    }
#endif

    // #633: THE RAM BEFORE THE ALLOCATION.  On Linux the arena is an anonymous mapping that succeeds whatever the host
    // has; its pages are committed as the load writes them, so a container whose memory limit is below the arena was
    // killed by the OOM killer part-way through the load, without a message.  A hard cgroup limit below it is
    // certain to end that way: refused, with the numbers.  Less RAM available than the arena (another program, an
    // engine still exiting) is only a warning - the OS may make room - per the recommend-not-force rule.
    ram_warning_.clear();
    // (a shared arena, --shared-expert-arena, may already be in RAM for another engine: not checked)
    if (detail::HostMemory hm; shared_arena_file.empty() && detail::host_available_memory(hm)) {
        const uint64_t need = want + (uint64_t) blob;
        const double gib = 1073741824.0;
        char buf[512];
        if (hm.cgroup_limit < need) {
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: the expert arena needs %.2f GiB of RAM but this process's memory limit "
                          "(cgroup memory.max / memory.limit_in_bytes) is %.2f GiB: it would be killed while loading. "
                          "Raise the container's limit, or run with less RAM: --mmap-experts with "
                          "--resident-budget-gib N keeps only the hottest experts in RAM",
                          (double) need / gib, (double) hm.cgroup_limit / gib);
            err = buf;
            return false;
        }
        if (hm.available < need) {
            std::snprintf(buf, sizeof buf,
                          "the expert arena needs %.2f GiB of RAM but %.2f GiB is available (%.2f GiB short): the load "
                          "may swap or be stopped by the OS. Close other programs (or wait for an engine that is "
                          "still exiting), or run with less RAM: --mmap-experts with --resident-budget-gib N",
                          (double) need / gib, (double) hm.available / gib, (double) (need - hm.available) / gib);
            ram_warning_ = buf;
        }
    }

    uint64_t pack_hash = 0;
    if (!shared_arena_file.empty() &&
        !shared_arena_pack_hash(pack_dir, path, gguf_, lay, pack_hash, err)) return false;

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    // Small-RAM machines: a native pack's experts (tens of GiB) as a file-backed mmap arena.  The first run
    // materializes `<pack_dir>/experts-native.bin` from the GGUF shards; later runs page in lazily and the
    // kernel page cache holds whatever fits - the same contract as FileExpertSource's mmap of experts.bin,
    // for a pack whose blobs are not the canonical Q2_0 size.
    if (from_gguf && file_backing_) {
#if defined(_WIN32)
        err = "ArenaExpertSource: file-backed native experts need the POSIX mmap path";
        return false;
#else
        const std::string fb = pack_dir + "/experts-native.bin";
        const uint64_t cap = want + (uint64_t) blob;
        bool preexisting = false;
        {
            std::ifstream f(fb, std::ios::binary | std::ios::ate);
            // a run killed mid-materialization leaves the full-size but INCOMPLETE file (ftruncate up front),
            // so trust it only with the completion marker written after the load finished
            if (f) {
                std::ifstream done(fb + ".done");
                preexisting = ((uint64_t) f.tellg() == cap) && done.good();
            }
        }
        file_fd_ = ::open(fb.c_str(), O_RDWR | O_CREAT, 0644);
        if (file_fd_ < 0) { err = "ArenaExpertSource: cannot open " + fb; return false; }
        // R13: the O_DIRECT twin fd.  If the filesystem refuses O_DIRECT (or STRATA_NO_ODIRECT=1), every
        // read site falls back to the buffered pread path unchanged.
        direct_ok_ = (std::getenv("STRATA_NO_ODIRECT") == nullptr);
        if (direct_ok_) {
            direct_fd_ = ::open(fb.c_str(), O_RDONLY | O_DIRECT);
            if (direct_fd_ < 0) { direct_ok_ = false; }
            else {
                // R20: learn the alignment this file truly needs instead of assuming 4096.
                struct stat sb;
                const uint64_t fb_bytes = (::fstat(direct_fd_, &sb) == 0 && sb.st_size > 0)
                                              ? (uint64_t) sb.st_size : file_map_bytes_;
                direct_align_ = probe_direct_alignment(direct_fd_, fb_bytes);
            }
        } else {
            direct_fd_ = -1;
        }
        if (!preexisting && ::ftruncate(file_fd_, (off_t) cap) != 0) {
            ::close(file_fd_); file_fd_ = -1;
            err = "ArenaExpertSource: cannot size " + fb;
            return false;
        }
        void* m = ::mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
        if (m == MAP_FAILED) {
            ::close(file_fd_); file_fd_ = -1;
            err = "ArenaExpertSource: mmap of " + fb + " failed (" + std::to_string(cap) + " B)";
            return false;
        }
        file_map_ = m;
        file_map_bytes_ = cap;
        base_ = (const uint8_t*) m;
        arena_ = nullptr;
        pinned_bytes_ = 0;
        slice_bytes_ = 0;
        dev_slice_.clear();
        note_ = "file-backed mmap " + fb +
                (preexisting ? " (pre-existing; lazy page-in)" : " (materializing from the GGUF shards)") +
                (direct_ok_ ? "; O_DIRECT expert reads ON" : "; O_DIRECT expert reads off");
        LoadStats st;
        if (preexisting) {
            st.bytes = want;
            st.layers = (uint64_t) n_layers;
        } else {
            st = load_experts_gguf(gguf_, (uint8_t*) m, lay, threads);
        }
        if (st.bytes != want) {
            close();
            err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
            return false;
        }
        blobs_ = n_layers * n_expert;
        n_expert_ = n_expert;
        reads_ = 0;
        gib_per_s_ = st.gib_per_second();
        if (!preexisting) {
            std::ofstream m(fb + ".done");
            m << "ok";
        }
        return true;
#endif
    }
    PinnedArena* a = new PinnedArena(want + (uint64_t) blob, bounds);
    PinnedArena* a = new PinnedArena(want + (uint64_t) blob, bounds, max_pinned_bytes,
                                     shared_arena_file, pack_hash);
    if (!a->valid()) {
        const std::string why = a->note;
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" +
              std::to_string(want + (uint64_t) blob) + " B)" +
              (why.empty() ? std::string{} : ": " + why);
        return false;
    }
    // #285: unbuffered when the drive is read anyway and the file cache could not keep the experts for the next
    // start either (a 64 GB PC); otherwise the buffered readers, which a warm restart serves from the cache
    std::vector<std::string> files;
    if (from_gguf) {
        for (int64_t l = 0; l < n_layers; ++l)
            for (int r = 0; r < 3; ++r) {
                const std::string f = expert_gguf_file(gguf_, lay, l, r);
                if (std::find(files.begin(), files.end(), f) == files.end()) files.push_back(f);
            }
    } else {
        files.push_back(path);
    }
    std::string why;
    const bool unbuffered = experts_unbuffered(files, want + (uint64_t) blob, why);
    const int readers = unbuffered ? std::max(threads, 16) : threads;   // 16 keep a PCIe 5 drive's queue full
    LoadStats st;
    if (from_gguf) {
        st = load_experts_gguf(gguf_, a->data(), lay, readers, unbuffered);
    } else {
        if (unbuffered) st = load_experts_direct(path, a->data(), loff, lbytes, readers, /*chunk=*/8u << 20);
        if (!unbuffered || (!st.ok && st.error.empty()))   // unaligned ranges: the buffered reader
            st = load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20);
    }
    std::fprintf(stderr, "strata generate: expert arena read %s (%s)\n", unbuffered ? "unbuffered" : "through the file cache",
                 why.c_str());
    if (!st.ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    // the first start: write experts.bin for the mapped starts after this one - only when the drive has room for it
    // and 2 GiB more (a full drive fails other writes too); otherwise this start says so and runs pinned as before
    std::error_code space_ec;
    const uint64_t free_disk = arena_mmap && from_gguf ? (uint64_t) std::filesystem::space(
        std::filesystem::path(path).parent_path(), space_ec).available : 0;
    const bool room = !space_ec && free_disk >= want + (uint64_t) blob + (2ull << 30);
    if (arena_mmap && from_gguf && !room)
        std::fprintf(stderr, "strata generate: STRATA_ARENA_MMAP: NOT writing %s - %.1f GiB free on that drive, it needs "
                             "%.1f GiB plus 2 GiB to spare; this start keeps the arena in RAM\n", path.c_str(),
                     (double) free_disk / 1073741824.0, (double) (want + (uint64_t) blob) / 1073741824.0);
    if (arena_mmap && from_gguf && room) {
        const std::string tmp = path + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        bool ok = f != nullptr && std::fwrite(a->data(), 1, (size_t) want, f) == (size_t) want;
        if (ok) {
            std::vector<uint8_t> pad((size_t) blob, 0);
            ok = std::fwrite(pad.data(), 1, pad.size(), f) == pad.size();
        }
        if (f) ok = std::fclose(f) == 0 && ok;
        if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) std::remove(tmp.c_str());
        std::fprintf(stderr, "strata generate: STRATA_ARENA_MMAP: %s %s (the next start maps it)\n",
                     ok ? "wrote" : "could NOT write", path.c_str());
    }
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
    return true;
}

void ArenaExpertSource::close() {
    if (pf_started_) {
        pf_stop_.store(true, std::memory_order_release);
        pf_epoch_.fetch_add(1, std::memory_order_release);
        for (auto& t : pf_workers_) t.join();
        pf_workers_.clear();
        pf_started_ = false;
        pf_stop_.store(false, std::memory_order_release);
    }
    ring_.clear();
    ring_of_.clear();
    ring_layer_ = -1;
    if (hot_arena_ != nullptr) {
#if !defined(_WIN32)
        if (hot_locked_ && hot_cap_ > 0) ::munlock(hot_arena_, (size_t) hot_cap_);
        ::munmap(hot_arena_, (size_t) hot_cap_);
#endif
        hot_arena_ = nullptr;
        hot_cap_ = hot_used_ = 0;
        hot_count_ = 0;
        hot_locked_ = false;
        hot_slot_.clear();
        dc_slot_bytes_ = 0;
        dc_slots_ = 0;
        dc_head_ = dc_tail_ = -1;
        dc_prev_.clear(); dc_next_.clear(); dc_idx_.clear(); dc_epoch_.clear();
        dc_free_.clear(); dc_admit_list_.clear();
        dc_admits_ = dc_evicts_ = dc_fallbacks_ = 0;
    }
    if (file_map_ != nullptr) {
#if !defined(_WIN32)
        if (file_map_bytes_ > 0) ::munmap(file_map_, file_map_bytes_);
#endif
        file_map_ = nullptr;
        file_map_bytes_ = 0;
    }
    if (file_fd_ >= 0) {
#if !defined(_WIN32)
        ::close(file_fd_);
#endif
        file_fd_ = -1;
    }
    if (direct_fd_ >= 0) {
#if !defined(_WIN32)
        ::close(direct_fd_);
#endif
        direct_fd_ = -1;
        direct_ok_ = false;
    }
#if !defined(_WIN32)
    if (map_ != nullptr) munmap(map_, (size_t) map_bytes_);
#endif
    map_ = nullptr;
    map_bytes_ = 0;
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

void ArenaExpertSource::prefetch(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    (void) layer; (void) expert;
#else
    if (map_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t off = lay.blob_offset(layer, expert) & ~(uint64_t) 4095;
    const uint64_t end = lay.blob_offset(layer, expert) + lay.blob_bytes(layer);
    madvise((uint8_t*) map_ + off, (size_t) (end - off), MADV_WILLNEED);
#endif
}

// Only the pages wholly inside the blob: a page shared with a neighbour the CPU may still read stays.  MADV_DONTNEED
// unmaps them from this process; they stay in the page cache as clean, unmapped pages - the first the kernel takes
// back under pressure, and a minor fault away when the CPU needs the expert again (an adaptive swap evicts it).
// Dropping them from the cache too (POSIX_FADV_DONTNEED) made every eviction a re-read from the disk: ~200 major
// faults a second in decode, +8 ms a window.  The memory the arena used to hold was never the cache itself but
// ROCclr's pin-in-place locks on it (see main) - locked pages are the ones the kernel cannot take back.
uint64_t ArenaExpertSource::release(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    (void) layer; (void) expert;
    return 0;
#else
    if (map_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t off = (lay.blob_offset(layer, expert) + 4095) & ~(uint64_t) 4095;
    const uint64_t end = (lay.blob_offset(layer, expert) + lay.blob_bytes(layer)) & ~(uint64_t) 4095;
    if (end <= off) return 0;
    if (madvise((uint8_t*) map_ + off, (size_t) (end - off), MADV_DONTNEED) != 0) return 0;
    return end - off;
#endif
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || layer >= strata::kernels::cpu::expert_layout().n_layers ||
        expert < 0 || expert >= n_expert_)
        return false;
    // R9b: a TIER-RESIDENT blob's host pointer is inside the cudaHostRegistered tier - safe for async DMA.
    // The in-flight ring blobs are NOT (their bytes land in the tier only when `wait_layer` commits), and the
    // file mapping never is.  `blob()` answers tier residents first, so this predicate agrees with the pointer
    // `blob()` returns for exactly the blobs that can be staged to the GPU.
    if (file_map_ != nullptr) {
        // R22: a blob is DMA-able when its slot's SLICE still has a live registration (the governor may have
        // shed the slice: unregistered pages are reclaimable, and a DMA over them would read zero-fill or
        // garbage); hot_slot_ alone is no longer the whole truth.
        if (hot_reg_.empty() || hot_slot_.empty()) return false;
        const int64_t idx = layer * n_expert_ + expert;
        if (idx < 0 || idx >= blobs_ || hot_slot_[(size_t) idx] < 0) return false;
        // the offset is slot_id * dc_slot_bytes_; the slice id is slot_id / hot_slice_slots_
        return hot_reg_[(size_t) ((uint64_t) hot_slot_[(size_t) idx] / (hot_slice_slots_ * dc_slot_bytes_))].load(
            std::memory_order_acquire);
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    // R9b: the registered tier's device alias (pcie-mode direct/kernel).  Only the single-slice form is
    // wired; multi-slice leaves `hot_dev_base_` null and DMA mode (the default for native packs) instead.
    if (file_map_ != nullptr) {
        if (!hot_dev_ok_ || hot_dev_base_ == nullptr || hot_slot_.empty()) return nullptr;
        if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
        const int64_t idx = layer * n_expert_ + expert;
        if (idx < 0 || idx >= blobs_ || hot_slot_[(size_t) idx] < 0) return nullptr;
        return hot_dev_base_ + hot_slot_[(size_t) idx];
    }
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

// ================================ THE PINNED HOT TIER (small-RAM machines) =========================//
// See the header.  Two things make this pay rather than duplicate the mapping:
//
//   * the tier is ANONYMOUS and `mlock`ed, so it is not a reclaim candidate.  The header above says the mapped
//     arena loses exactly that property ("file-backed pages are the ones the OS drops from the standby list"),
//     and measured 42.8 GB/s reading anonymous memory against ~19 GB/s through the mapping;
//   * each blob is copied with ONE whole-blob read out of the mapping.  A 2.6 MB blob touched a 4 KiB page at a
//     time costs ~340 us per fault on this class of drive (~45x the drive's own 3.8 ms for the whole blob).
//
// The order is the PROFILE's, so the tier holds the highest-frequency blobs first and what still misses is the
// tail of the distribution.  After each range is copied its file pages are released with POSIX_FADV_DONTNEED:
// they will never be read from the file again (this tier serves them), and on a 30 GiB box the 10-14 GiB that
// frees is what lets the COLD expert set stay cached for prefill's full pass over it.
void ArenaExpertSource::pin_hot(const std::vector<std::pair<int32_t, int32_t>>& ranked, uint64_t bytes) {
    if (base_ == nullptr || file_map_ == nullptr || bytes == 0 || ranked.empty() || blobs_ <= 0 || hot_arena_ != nullptr)
        return;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_expert <= 0 || lay.n_layers <= 0) return;
    // R8: the tier is dynamic (LRU with profile-seeded contents) unless the A/B arm asks for the static form.
    dynamic_tier_ = (std::getenv("STRATA_STATIC_TIER") == nullptr);
#if !defined(_WIN32)
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
    // R7: **MADV_HUGEPAGE, AND READING STRAIGHT INTO THE TIER.**  Two changes from the R5 form, both measured
    // on the reason the tier exists:
    //
    //   * TLB.  The whole point of this tier is that a decode step walks ~460 two-megabyte blobs at random.
    //     Through 4 KiB pages that is ~550,000 TLB entries of working set per window; through 2 MiB pages it is
    //     ~460.  `MAP_HUGETLB` cannot be used - this box has 0 reserved hugepages and reserving GiB of them
    //     would fight the kernel for the same RAM - but THP is `always` here with `defer+madvise` defrag, so
    //     `MADV_HUGEPAGE` on the anonymous range is what actually gets the 2 MiB pages.  The mapping is sized
    //     UP to a 2 MiB multiple and the blob starts are 256-byte aligned, so no blob straddles a boundary
    //     that matters.
    //   * I/O.  The R5 form memcpy'd out of the file MAPPING, so filling a 12-20 GiB tier faulted 4 KiB at a
    //     time and left 12-20 GiB of useless page cache behind.  It now `pread`s whole blobs into the tier
    //     directly: one syscall per blob at the drive's own rate, and nothing is left in the page cache for
    //     the kernel to evict the hot set in favour of.
    //
    // R8: the arena is carved into FIXED SLOTS of the largest blob (4 KiB-rounded).  Blob sizes are uniform
    // across this pack's layers to ~1% (measured: 5,318 disk blobs / 11.57 GB = 2.176 MB average against a
    // 2.18 MB max), so the per-slot waste is negligible, and fixed slots make admission, eviction and the LRU
    // O(1) array walks - no allocator, no fragmentation, and a freed slot holds any layer's next blob.
    const uint64_t slot = ((uint64_t) lay.max_blob + 4095u) / 4096u * 4096u;
    const uint64_t cap = bytes / slot * slot;
    if (cap == 0) { note_ += "; hot tier skipped (bytes below one slot)"; return; }
    void* m = ::mmap(nullptr, (size_t) cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        note_ += "; hot tier NOT allocated (mmap of " + std::to_string(cap) + " B failed)";
        return;
    }
    hot_arena_ = (uint8_t*) m;
    hot_cap_ = cap;
    const bool huge = (::madvise(m, (size_t) cap, MADV_HUGEPAGE) == 0);
    dc_init((int64_t) slot);
    hot_slot_.assign((size_t) blobs_, -1);
    for (const auto& pr : ranked) {
        if (hot_count_ >= dc_slots_) break;   // the arena is full of whole slots
        const int32_t l = pr.first, e = pr.second;
        if (l < 0 || e < 0 || l >= lay.n_layers || e >= lay.n_expert) continue;
        const int64_t idx = (int64_t) l * n_expert_ + (int64_t) e;
        if (idx < 0 || idx >= blobs_ || hot_slot_[(size_t) idx] >= 0) continue;
        const uint64_t len = lay.blob_bytes(l);
        const uint64_t off = lay.blob_offset(l, e);
        if (len == 0 || off + len > file_map_bytes_ || len > dc_slot_bytes_) break;
        uint8_t* dst = hot_arena_ + (uint64_t) hot_count_ * dc_slot_bytes_;
        bool ok = false;
        bool used_direct = false;
        if (file_fd_ >= 0) {
            // R13: O_DIRECT first; the buffered fill (with its DONTNEED page-cache drop) is the fallback.
            used_direct = direct_read_span(off, len, dst);
            ok = used_direct;
            if (!ok) {
                uint64_t done = 0;
                ok = true;
                while (done < len) {
                    const ssize_t r = ::pread(file_fd_, dst + done, (size_t) (len - done), (off_t) (off + done));
                    if (r <= 0) { ok = false; break; }
                    done += (uint64_t) r;
                }
            }
        } else {
            std::memcpy(dst, base_ + off, (size_t) len);
            ok = true;
        }
        if (!ok) break;
        // R8: rank order IS the initial LRU order - rank 0 at the head (evicted last), the rank cutoff at
        // the tail, so a domain shift evicts the coldest profile blobs first.
        const int32_t s = (int32_t) hot_count_;
        dc_idx_[(size_t) s] = (int32_t) idx;
        dc_epoch_[(size_t) s] = 0;
        // R8.1: the profile seeds the frequency dimension too.  64 halves to ~1 after six untouched windows,
        // so on a genuinely different domain the cold profile blobs DO yield - but a one-shot miss flood
        // (count 1) can never flush them within a single request.
        dc_count_[(size_t) s] = 64;
        dc_prev_[(size_t) s] = dc_tail_;
        dc_next_[(size_t) s] = -1;
        if (dc_tail_ >= 0) dc_next_[(size_t) dc_tail_] = s; else dc_head_ = s;
        dc_tail_ = s;
        hot_slot_[(size_t) idx] = (int64_t) ((uint64_t) s * dc_slot_bytes_);
        ++hot_count_;
        // **AND GIVE THE FILE PAGES BACK.**  A `pread` of a blob leaves that blob in the page cache, so filling
        // a 20 GiB tier would otherwise leave 20 GiB of page cache behind - for pages this tier now serves from
        // anonymous memory and will never read from the file again.  On a 30 GiB box that page cache is the
        // difference between a hot tier that can be grown and one that cannot.  (R13: the O_DIRECT fill leaves
        // no cache copy, so the fadvise only runs on the buffered fallback.)
        if (file_fd_ >= 0 && !used_direct) ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
    }
    for (int64_t s = hot_count_; s < dc_slots_; ++s) dc_free_.push_back((int32_t) s);
    if (hot_count_ == 0) {
        ::munmap(hot_arena_, (size_t) hot_cap_);
        hot_arena_ = nullptr; hot_cap_ = 0; hot_slot_.clear();
        note_ += "; hot tier empty";
        return;
    }
    hot_used_ = (uint64_t) hot_count_ * dc_slot_bytes_;
    // R8: lock AFTER the fill (the R7 order this kernel accepts).  Locking the untouched mapping BEFORE
    // filling fails with ENOMEM here (22 GiB of THP prefault under fragmentation) and the failure mode is
    // catastrophic: the tier runs reclaimable and the kernel evicts hot-tier pages mid-generation.  mlock()
    // sets VM_LOCKED on the whole VMA, so pages faulted LATER by an LRU admission inside the range are
    // locked too - admissions need nothing extra.
    hot_locked_ = (::mlock(hot_arena_, (size_t) hot_cap_) == 0);
    if (!dynamic_tier_ && hot_used_ < hot_cap_) {   // static A/B: the free tail has no future, give it back
        ::munmap(hot_arena_ + hot_used_, (size_t) (hot_cap_ - hot_used_));
        hot_cap_ = hot_used_;
        dc_slots_ = (int64_t) hot_count_;
    }
    if (!dynamic_tier_) hot_locked_ = (::mlock(hot_arena_, (size_t) hot_used_) == 0);
    // ---- R9b: REGISTER THE TIER WITH CUDA so the GPU's PCIe expert path can read it.
    //
    // With `--mmap-experts` the arena is a plain file mmap: `pinned_bytes_` is 0, `dev_slice_` empty, so the
    // verify plan builder's `pinned()`/`device_alias()` gates were permanently closed and EVERY routed expert
    // was computed on the CPU pool - measured ~330 ms of serial pool work per verify window on this box, which
    // is the decode wall all by itself.  The tier is anonymous, mlocked and 2 MiB-page-backed - exactly what
    // `cudaHostRegister` wants - and tier-resident blobs are 97%+ of routed traffic, so registering it turns
    // `fetch_dma` (default --pcie-mode auto -> DMA for native packs, --pcie-frac 0.55) back on: the copy
    // engine moves the layer's PCIe share into VRAM staging beside the CPU's own work and the grouped kernel
    // computes it on the GPU.
    //
    // R22: **REGISTER IN SLICES, NOT ONE WHOLE RANGE.**  Registering pins the pages (unswappable), which is
    // exactly the over-subscription that kills a long-horizon box at deep context (the 24 GiB tier + KV pools
    // + desktop left no headroom: zram collapse, then cublasCreate died inside the driver - 2026-10-04/05
    // twice).  Slices of whole slots (12 per 24 GiB tier, ~2 GiB) get their own registration so the pressure
    // governor (`shed_pressure`) can UNREGISTER the cold-end slices first, drop their pages and put their
    // slots back on the free list: the tier SHRINKS under pressure and refills when it passes, instead of
    // being pinned until the OOM.  Only the direct/kernel alias form (a single whole-range `hot_dev_base_`,
    // pcie-mode direct) is lost - the served mode is DMA (auto -> DMA for native packs), which needs only
    // `pinned()` per blob, and slice-shaped registrations can no longer hand out a single alias pointer.
    if (hot_locked_ && hot_arena_ != nullptr && hot_cap_ > 0 && file_map_ != nullptr) {
        auto reg = [&](uint8_t* p, uint64_t n) {
            return cudaHostRegister(p, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped);
        };
        const int64_t ideal_slices = 12;
        hot_slice_slots_ = (dc_slots_ + ideal_slices - 1) / ideal_slices;   // whole slots per ~2 GiB slice
        const int64_t nslice = (dc_slots_ + hot_slice_slots_ - 1) / hot_slice_slots_;
        hot_reg_ = std::vector<std::atomic<bool>>((size_t) nslice);   // each slice's flag starts false
        hot_dev_ok_ = true;
        hot_dev_base_ = nullptr;
        for (int64_t i = 0; i < nslice; ++i) {
            const int64_t s0 = i * hot_slice_slots_;
            const uint64_t off = (uint64_t) s0 * dc_slot_bytes_;
            const uint64_t n = std::min((uint64_t) hot_slice_slots_ * dc_slot_bytes_, hot_cap_ - off);
            if (reg(hot_arena_ + off, n) == cudaSuccess) {
                hot_reg_[(size_t) i].store(true, std::memory_order_relaxed);
            } else {
                (void) cudaGetLastError();   // a slice that cannot register loses DMA for its blobs only:
                                             // pinned() gates per slice, and the pool computes them instead
            }
        }
    }
#endif
    char buf[420];
    std::snprintf(buf, sizeof buf,
                  "; hot tier %.2f GiB in %lld blobs from the profile%s%s (THP %s, slot %.2f MiB, %lld slots, %lld free)",
                  (double) hot_used_ / 1073741824.0, (long long) hot_count_,
                  hot_locked_ ? " (mlocked)" : " (mlock FAILED - reclaimable)",
                  dynamic_tier_ ? ", LRU adaptive" : ", static", huge ? "on" : "off",
                  (double) dc_slot_bytes_ / 1048576.0, (long long) dc_slots_,
                  (long long) dc_free_.size());
    note_ += buf;
    if (file_map_ != nullptr)
        note_ += hot_dev_ok_ ? (hot_dev_base_ ? "; tier cudaHostRegistered+device-alias (GPU PCIe expert path ON)"
                                              : "; tier cudaHostRegistered sliced (GPU PCIe DMA path ON)")
                             : "; tier NOT registered (GPU PCIe expert path off)";
}

// ---- R8: the adaptive LRU tier ---------------------------------------------------------------------------

// R22: THE PRESSURE GOVERNOR'S SHED.  See the header.  Called from the serve loop's line handler between
// requests, so no decode window is active and no reads are in flight; the epoch guard still refuses a slice
// whose slots were touched or admitted this window (a late commit or a stray caller would otherwise see
// zero-fill memory where its bytes used to be).
//
// R22b AUDIT: the shed order is SCORE-BASED, not positional.  The "arena END = the coldest" assumption only
// holds for a FRESH tier: the decode-side LRU (dc_alloc runs from begin_layer even in the static tier)
// recycles slot CONTENT over a long session, so the tail slices hold an arbitrary mix and a positional shed
// could drop the session's actual working set - the exact speed cliff this feature was built to avoid.  Each
// slice is scored by its slots' frequency counts plus a freshness bonus for anything touched this or the
// previous window (decode patterns repeat windows: `win_repeat`), and the coldest scored slice is shed
// first.  Ties break toward the arena's end (old profile placements).
uint64_t ArenaExpertSource::shed_pressure(uint64_t bytes_to_free) {
    if (hot_arena_ == nullptr || hot_slot_.empty() || dc_slot_bytes_ == 0 || hot_slice_slots_ <= 0) return 0;
    const int64_t nslice = (int64_t) hot_reg_.size();
    if (nslice <= 1) return 0;                      // never shed the tier's single (hottest) slice
    uint64_t freed = 0;
    std::lock_guard<std::mutex> lk(dc_mu_);
    std::vector<int64_t> eligible_present;
    eligible_present.clear();
    // eligible = slices with a live registration whose slots are not part of this window's traffic
    for (int64_t i = nslice - 1; i > 0; --i) {
        if (!hot_reg_[(size_t) i].load(std::memory_order_relaxed)) continue;
        const int64_t s0 = i * hot_slice_slots_;
        bool safe = true;
        for (int64_t s = s0; s < dc_slots_ && s < s0 + hot_slice_slots_; ++s) {
            if (dc_idx_[(size_t) s] < 0) continue;
            const uint32_t e = dc_epoch_[(size_t) s];
            if (e == window_epoch_ || (e > 0 && e == window_epoch_ - 1)) { safe = false; break; }
        }
        if (safe) eligible_present.push_back(i);
    }
    while (freed < bytes_to_free && !eligible_present.empty()) {
        // score every eligible slice: the sum of its slots' frequency credits (a 0-count slot counts 1 -
        // never cheaper than a warm slot, since the profile placements under it have rank value), then pick
        // the lowest score; ties break toward the higher index (the old profile tail).  Freshness is folded
        // into eligibility above; nothing can move under dc_mu_ while we score, but the entry is re-checked
        // by exactly the same read that built the eligibility list.
        int64_t best_i = -1;
        uint64_t best_score = ~0ull;
        for (const int64_t i : eligible_present) {
            const int64_t s0 = i * hot_slice_slots_;
            uint64_t score = 0;
            for (int64_t s = s0; s < dc_slots_ && s < s0 + hot_slice_slots_; ++s)
                score += dc_count_[(size_t) s] ? (uint64_t) dc_count_[(size_t) s] : 1ull;
            if (score < best_score || (score == best_score && i > best_i)) { best_score = score; best_i = i; }
        }
        if (best_i < 0) break;
        {
            const int64_t s0 = best_i * hot_slice_slots_;
            const uint64_t off = (uint64_t) s0 * dc_slot_bytes_;
            const uint64_t n = std::min((uint64_t) hot_slice_slots_ * dc_slot_bytes_, hot_cap_ - off);
            if (n == 0) break;
            cudaHostUnregister(hot_arena_ + off);            // the pointer must match the registration's own
            (void) cudaGetLastError();
            ::munlock(hot_arena_ + off, (size_t) n);        // the whole-arena mlock splits per range at munlock
            (void) ::madvise(hot_arena_ + off, (size_t) n, MADV_DONTNEED);   // drop the pages: the RAM is given back
            hot_reg_[(size_t) best_i].store(false, std::memory_order_release);
            for (int64_t s = s0; s < dc_slots_ && s < s0 + hot_slice_slots_; ++s) {
                const int64_t vi = dc_idx_[(size_t) s];
                if (vi < 0) continue;          // already free: pushing it again would double-allocate it
                if (vi < (int64_t) hot_slot_.size()) {
                    hot_slot_[(size_t) vi] = -1;
                    dc_unlink((int32_t) s);
                }
                dc_idx_[(size_t) s] = -1;
                dc_count_[(size_t) s] = 0;
                dc_epoch_[(size_t) s] = 0;
                dc_free_.push_back((int32_t) s);
            }
            hot_shed_ += n;
            freed += n;
        }
        {   // drop the winner from the eligible list; unregistered slices cannot be re-scored
            std::vector<int64_t> keep;
            keep.reserve(eligible_present.size());
            for (const int64_t i : eligible_present) if (i != best_i) keep.push_back(i);
            eligible_present.swap(keep);
        }
    }
    return freed;
}

// R22b: THE GOVERNOR'S REGROW.  See the header.  The server sends `REGROW <bytes>` only when ITS reading of
// MemAvailable is comfortably above the floor (>= floor + 2.5 GiB), which together with this precheck
// (MemAvailable >= slice + 1024 MiB) gives the regrow a double hysteresis: the tier can never regrow into
// the pressure that made it shed.
uint64_t ArenaExpertSource::regrow_pressure(uint64_t bytes) {
    if (hot_arena_ == nullptr || hot_reg_.empty() || dc_slot_bytes_ == 0 || hot_slice_slots_ <= 0) return 0;
    const int64_t nslice = (int64_t) hot_reg_.size();
    int64_t target = -1;
    for (int64_t i = nslice - 1; i > 0; --i) {      // the arena's END slices first: the LIFO free list pops the
        if (!hot_reg_[(size_t) i].load(std::memory_order_relaxed)) { target = i; break; }   // high slots first
    }
    if (target < 0) return 0;                       // nothing is shed; no regrow needed
    const int64_t s0 = target * hot_slice_slots_;
    const uint64_t off = (uint64_t) s0 * dc_slot_bytes_;
    const uint64_t n = std::min((uint64_t) hot_slice_slots_ * dc_slot_bytes_, hot_cap_ - off);
    if (n == 0) return 0;
    if (bytes > 0 && n > bytes) return 0;           // the caller budgets per call; one slice at a time
    const uint64_t avail = strata::platform::host_available_bytes();
    if (avail < n + ((uint64_t) 1024 << 20)) return 0;   // the regrow faults the whole slice back at once
    std::lock_guard<std::mutex> lk(dc_mu_);
    if (cudaHostRegister(hot_arena_ + off, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped)
        != cudaSuccess) {
        (void) cudaGetLastError();     // refuse the regrow rather than half-pinning: the next cycle retries
        return 0;
    }
    ::mlock(hot_arena_ + off, (size_t) n);             // VM_LOCKED splits per range, as it did at pin_hot
    hot_reg_[(size_t) target].store(true, std::memory_order_release);
    hot_shed_ = hot_shed_ > n ? hot_shed_ - n : 0;
    return n;
}

void ArenaExpertSource::dc_init(int64_t slot_bytes) {
    dc_slot_bytes_ = (uint64_t) slot_bytes;
    dc_slots_ = hot_cap_ > 0 ? (int64_t) (hot_cap_ / dc_slot_bytes_) : 0;
    dc_prev_.assign((size_t) dc_slots_, -1);
    dc_next_.assign((size_t) dc_slots_, -1);
    dc_idx_.assign((size_t) dc_slots_, -1);
    dc_epoch_.assign((size_t) dc_slots_, 0);
    dc_count_.assign((size_t) dc_slots_, 0);
    dc_free_.clear();
    dc_head_ = dc_tail_ = -1;
    dc_admit_list_.clear();
    dc_admits_ = dc_evicts_ = dc_fallbacks_ = 0;
}

void ArenaExpertSource::dc_unlink(int32_t s) {
    // R22: a slot in `dc_free_` is DETACHED (prev = next = -1, head != s): touching the -1 sentinels here used
    // to set head = tail = -1 and silently collapse the LRU to one slot (the double-unlink bug the alloc
    // comment warns about).  Shed slots join the free list, so allocations from it now hit this path: detached
    // nodes are skipped, the list stays intact.
    const int32_t p = dc_prev_[(size_t) s], n = dc_next_[(size_t) s];
    if (p < 0 && n < 0 && dc_head_ != (int32_t) s) return;
    if (p >= 0) dc_next_[(size_t) p] = n; else dc_head_ = n;
    if (n >= 0) dc_prev_[(size_t) n] = p; else dc_tail_ = p;
    dc_prev_[(size_t) s] = dc_next_[(size_t) s] = -1;
}

void ArenaExpertSource::dc_link_head(int32_t s) {
    dc_prev_[(size_t) s] = -1;
    dc_next_[(size_t) s] = dc_head_;
    if (dc_head_ >= 0) dc_prev_[(size_t) dc_head_] = s;
    dc_head_ = s;
    if (dc_tail_ < 0) dc_tail_ = s;
}

void ArenaExpertSource::dc_touch(int32_t s) {
    std::lock_guard<std::mutex> lk(dc_mu_);
    dc_touch_locked(s);
}

void ArenaExpertSource::dc_touch_locked(int32_t s) {
    if (s < 0 || s >= (int32_t) dc_slots_) return;
    if (dc_head_ != s) {
        dc_unlink(s);
        dc_link_head(s);
    }
    dc_epoch_[(size_t) s] = window_epoch_;
    // R8.1: frequency.  A one-shot miss (a diverse generation's novel expert) stays cheap to evict; a blob
    // the workload keeps re-referencing gets expensive to evict.  Capped so a long-lived favourite cannot
    // become untouchable.
    if (dc_count_[(size_t) s] < 4095) ++dc_count_[(size_t) s];
}

int32_t ArenaExpertSource::dc_alloc(int64_t idx) {
    std::lock_guard<std::mutex> lk(dc_mu_);
    return dc_alloc_locked(idx);
}

int32_t ArenaExpertSource::dc_alloc_locked(int64_t idx) {
    int32_t s;
    if (!dc_free_.empty()) {
        s = dc_free_.back();
        dc_free_.pop_back();
    } else {
        // R8.1: EVICT BY FREQUENCY, WITH LRU AS THE TIE-BREAK.  The pure-LRU form thrashes on a diverse
        // long generation: every miss is a one-shot (measured: 9,754 misses, 0% window-to-window repeat),
        // so the flood admitted and evicted 15,914 blobs in one request and decode fell to 7.6 tok/s while
        // the resident profile content - which WOULD have been hit - was flushed.  The victim is now the
        // lowest-hit-count slot within a bounded walk from the LRU tail (LRU position breaks ties), so a
        // one-shot flood recycles among itself and never pushes out content the workload actually re-uses.
        s = -1;
        int32_t best = -1;
        uint16_t best_count = 0xffff;
        int32_t v = dc_tail_;
        for (int scanned = 0; v >= 0 && scanned < 512; ++scanned, v = dc_prev_[(size_t) v]) {
            if (dc_epoch_[(size_t) v] == window_epoch_) continue;   // this window's own bytes
            const uint16_t c = dc_count_[(size_t) v];
            if (c < best_count) { best_count = c; best = v; if (c == 0) break; }
        }
        if (best < 0) return -1;   // every slot is in use by this window: fall back to the legacy ring
        s = best;
        const int64_t vidx = dc_idx_[(size_t) s];
        if (vidx >= 0 && vidx < (int64_t) hot_slot_.size()) hot_slot_[(size_t) vidx] = -1;
        ++dc_evicts_;
        // NO explicit unlink here: `dc_touch_locked` below unlinks the victim (it is still in the list) and
        // relinks it at the head.  Unlinking twice corrupts the intrusive list - the second unlink reads the
        // -1 sentinels and sets head = tail = -1, which silently collapses the LRU to one slot and every
        // later allocation evicts nothing (measured: 394 admissions against 16,671 ring fallbacks).
    }
    dc_idx_[(size_t) s] = (int32_t) idx;
    dc_touch_locked(s);
    dc_count_[(size_t) s] = 1;   // admitted: one shot of credit until the workload proves otherwise
    return s;
}

// R8b: **PREFILL ADMISSIONS.**  See the header.  `bytes` are the staging buffer's landed copy of
// `(layer, expert)`; a resident blob just gets its frequency touch, a miss takes a slot (evicting the
// lowest-value victim) and memcpy's in.  Called from prefill's consume step, host thread, serialized against
// the decode path - but NOT against prefill's own reader threads touching resident blobs, hence the lock.
bool ArenaExpertSource::dc_stage_admit(int64_t layer, int64_t expert, const uint8_t* bytes) {
    if (dc_slots_ <= 0 || hot_slot_.empty() || bytes == nullptr) return false;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return false;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0 || len > dc_slot_bytes_) return false;
    std::lock_guard<std::mutex> lk(dc_mu_);
    // R22: THE STATIC TIER REFILLS THE GOVERNOR'S SHED.  A static tier (STRATA_STATIC_TIER) used to refuse
    // every runtime admission; that is still the case while all its slots are home - but after
    // `shed_pressure` frees slots for the host's RAM, the prefill staging is exactly the path that can put
    // the prompt's own working set back, so a static tier admits INTO THE FREE LIST only.  The check sits
    // INSIDE the tier lock on purpose: a reader that raced the empty-check past the lock could otherwise
    // fall into dc_alloc_locked's EVICT path (path B) and flush profile content - the one thing a static
    // tier must never do.  The R10 flood rules (count = 0) cap it: a shed tier cannot be refilled past
    // its own freed size.
    if (!dynamic_tier_ && dc_free_.empty()) return false;
    if (hot_slot_[(size_t) idx] >= 0) {          // resident: this prompt's evidence of reuse
        dc_touch_locked((int32_t) ((uint64_t) hot_slot_[(size_t) idx] / dc_slot_bytes_));
        return true;
    }
    const int32_t s = dc_alloc_locked(idx);
    if (s < 0) return false;
    std::memcpy(hot_arena_ + (uint64_t) s * dc_slot_bytes_, bytes, (size_t) len);
    hot_slot_[(size_t) idx] = (int64_t) ((uint64_t) s * dc_slot_bytes_);
    // R10: prefill admissions enter with ZERO credit, not one.  A big prompt's sweep admits ~24k blobs
    // against ~11.6k slots; with count=1 the flood tied the untouched profile blobs and the LRU
    // tie-break evicted the PROFILE side (it is always older), so the decode after a prefill lost its
    // hot set (measured: multi-topic median 7.4 tok/s with the flood, 8.4 with admissions off).  With
    // count=0 the flood recycles among itself unless a later touch proves the blob hot - the same
    // rule the decode ring earns its count=1 credit by.
    dc_count_[(size_t) s] = 0;
    ++dc_stage_admits_;
    return true;
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
    ++blob_stats_.requests;
    // R8: the hot tier is checked FIRST.  An admission is committed here in `wait_layer`, before the pass-1
    // `blob()` calls run - and a committed blob must NOT fall through to the ring check, because its ring
    // slot (when it has one at all) is the legacy staging buffer, not the memory its bytes were read into.
    if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
        ++hot_lookups_;
        ++hot_hits_;
        ++blob_stats_.hot;
        if (dc_slots_ > 0) dc_touch((int32_t) ((uint64_t) hot_slot_[(size_t) idx] / dc_slot_bytes_));
        return hot_arena_ + hot_slot_[(size_t) idx];
    }
    // R5-fetch: the ring serves THIS layer's pread misses (see begin_layer); `ring_layer_` pins the validity
    // to the layer the reads were issued for, so a later `blob()` (the profile fill, prefill's staging) can
    // never see a previous layer's bytes.
    if (layer == ring_layer_ && (int64_t) ring_of_.size() == blobs_ && ring_of_[(size_t) idx] >= 0) {
        ++blob_stats_.ring;
        return ring_[(size_t) ring_of_[(size_t) idx]].data();
    }
    // R7: the fall-through is a fault on the file mapping.  If the page is in the page cache it is a DRAM read;
    // if not, the kernel goes to the drive for it, 4 KiB at a time.  This is the path the R5 pread ring exists
    // to avoid, and counting it is how a run says whether it avoided it.
    ++blob_stats_.map;
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

ArenaExpertSource::BlobStats ArenaExpertSource::take_blob_stats() {
    BlobStats s = blob_stats_;
    blob_stats_ = BlobStats{};
    return s;
}

// R8: the stats-free core of `read_blob`, safe to call from several staging readers at once.  A tier-resident
// blob memcpy's out of the locked arena (read-only); anything else is one whole-blob `pread` out of the file.
int64_t ArenaExpertSource::read_blob_raw(int64_t layer, int64_t expert, void* dst) {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return -1;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return -1;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0) return -1;
    {   // R8b: the lookup, the copy and the touch hold `dc_mu_` TOGETHER, because the host thread's
        // `dc_stage_admit` can evict this slot between the lookup and the copy - a memcpy out of a freed
        // slot would hand the staging buffer torn bytes that reach the GPU as a plausible, wrong expert.
        std::lock_guard<std::mutex> lk(dc_mu_);
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
            std::memcpy(dst, hot_arena_ + hot_slot_[(size_t) idx], (size_t) len);
            dc_touch_locked((int32_t) ((uint64_t) hot_slot_[(size_t) idx] / dc_slot_bytes_));
            return (int64_t) len;
        }
    }
    if (file_fd_ < 0 || file_map_ == nullptr) return -1;
    const uint64_t off = lay.blob_offset(layer, expert);
    if (off + len > file_map_bytes_) return -1;
    // R13: O_DIRECT first (no page-cache alloc/copy/DONTNEED); buffered pread is the fallback arm.
    if (direct_read_span(off, len, (uint8_t*) dst)) return (int64_t) len;
    uint8_t* p = (uint8_t*) dst;
    uint64_t done = 0;
    while (done < len) {
        const ssize_t r = ::pread(file_fd_, p + done, (size_t) (len - done), (off_t) (off + done));
        if (r <= 0) return -1;
        done += (uint64_t) r;
    }
    // R8.2: same page-cache argument as the ring reader - the staging buffer holds the bytes, the cache
    // copy only fuels reclaim churn (see pf_worker).
    ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
    return (int64_t) len;
}

// R5-fetch: read one whole blob into `dst`.  The hot tier memcpy's out of the locked arena; the file-backed
// arena issues ONE sequential `pread`, which the drive serves at its full rate - against ~340 us of latency
// per 4 KiB page fault when the pool walks the mapping instead (see `pin_hot`'s comment for the measurement).
int64_t ArenaExpertSource::read_blob(int64_t layer, int64_t expert, void* dst) {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return -1;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return -1;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0) return -1;
    if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
        std::memcpy(dst, hot_arena_ + hot_slot_[(size_t) idx], (size_t) len);
        ++prefetched_;
        return (int64_t) len;
    }
    const int64_t r = read_blob_raw(layer, expert, dst);
    if (r < 0) return -1;
    ++prefetched_;
    ++blob_stats_.disk;
    blob_stats_.disk_bytes += (int64_t) len;
    return r;
}

// R5-fetch: the persistent reader threads.  The protocol is ExpertPool's (pool.cpp): a worker counts itself
// parked BEFORE its first wait so the host's publish below can never outrun a thread that is still leaving
// the previous batch's claim loop - a stolen or double-claimed job would make `pf_done_` overshoot the batch
// size and the host would wait forever.  R5b: the park is a BLOCKING condvar wait, not a spin - the readers
// idle most of each layer (the pool computes), and spinning threads on the SMT siblings measurably slow the
// compute workers down.
void ArenaExpertSource::pf_worker() {
    uint32_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(pf_mu_);
            pf_parked_.fetch_add(1, std::memory_order_acq_rel);   // arrive at the park before the first wait
            pf_cv_.wait(lk, [&] { return pf_epoch_.load(std::memory_order_acquire) != seen ||
                                         pf_stop_.load(std::memory_order_relaxed); });
            pf_parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park
        }
        if (pf_stop_.load(std::memory_order_acquire)) return;
        seen = pf_epoch_.load(std::memory_order_relaxed);
        for (;;) {
            const uint32_t i = pf_head_.fetch_add(1, std::memory_order_relaxed);
            if (i >= pf_njobs_.load(std::memory_order_acquire)) break;
            const PfJob& j = pf_jobs_[(size_t) i];
            // R13: the direct path first; the buffered pread + DONTNEED stays as the fallback arm.
            bool ok = direct_read_span(j.off, j.len, j.dst);
            if (!ok) {
                uint64_t done = 0;
                ok = true;
                while (done < j.len) {
                    const ssize_t r = ::pread(file_fd_, j.dst + done, (size_t) (j.len - done), (off_t) (j.off + done));
                    if (r <= 0) { ok = false; break; }
                    done += (uint64_t) r;
                }
                if (ok)
                    // R8.2: **DROP THE PAGE-CACHE COPY IMMEDIATELY.**  A whole-blob `pread` leaves ~530 cached pages
                    // behind, and on a 30 GiB box that already runs a 22 GiB mlocked tier there is no cache left to
                    // hold them: the kernel allocates, clears, copies and then RECLAIMS every page of every miss.
                    // perf during a diverse 300-token generation measured ~39% of the engine's CPU inside kernel
                    // page-management on exactly this cycle.  The bytes the ring needed are in the ring; the cache
                    // copy serves nobody (decode re-uses go through the LRU tier, prefill staging re-reads nothing).
                    ::posix_fadvise(file_fd_, (off_t) j.off, (off_t) j.len, POSIX_FADV_DONTNEED);
            }
            pf_done_.fetch_add(1, std::memory_order_release);
        }
    }
}

// R5-fetch.  Called with a layer's routing ids before the pool sets up its jobs.  The misses of the layer
// (what neither the hot tier nor a previous read can answer) are read WHOLE into the ring with `pf_workers_`
// concurrent preads - the drive streams them back-to-back at its sequential rate while the host finishes
// dispatching, instead of the drain faulting 4 KiB at a time and waiting on each page.  The ring is one
// layer deep: the dispatch contract has the pool block until this layer's drain is done, so the slots are
// free the next time this runs.  Blobs beyond the ring (more unique misses than slots - not reachable with
// this model's k=10 (+shared) and MAXT=4, but never say never) fall back to one WILLNEED each.
void ArenaExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    if (base_ == nullptr || ids == nullptr || k <= 0) return;
    if (layer < 0 || layer >= strata::kernels::cpu::expert_layout().n_layers) return;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if ((int64_t) pf_seen_.size() != n_expert_) pf_seen_.assign((size_t) n_expert_, 0);
    else std::fill(pf_seen_.begin(), pf_seen_.end(), (uint8_t) 0);
    if (!pf_started_ && file_fd_ >= 0 && file_map_ != nullptr) {
        // R7: the predictor is on by default; `STRATA_NO_PREDICT=1` is the A/B arm.
        // R8: off by default (see pf_window_start); STRATA_PREDICT=1 turns it back on.
        prefetch_predict_ = (std::getenv("STRATA_PREDICT") != nullptr);
        ring_.assign(kRingSlots, {});
        for (auto& r : ring_) r.resize((size_t) lay.max_blob + 512);
        ring_of_.assign((size_t) blobs_, -1);
        pf_jobs_.resize((size_t) kRingSlots * (size_t) kMaxChunks);
        for (int i = 0; i < 10; ++i) pf_workers_.emplace_back([this] { pf_worker(); });
        pf_started_ = true;
    }
    if (!pf_started_) {
        // resident arena or no file: nothing to fetch, but keep the dedupe+stats behaviour
        return;
    }
    // R7: a new pass starts at layer 0, and the previous pass's ring is not valid any more.  Without this the
    // `layer == ring_layer_` gate in `blob()` would serve a stale blob to anything that asks for the last
    // decode layer's index (the prefill suffix of a KEEP request is the caller that can reach it).
    if (layer == 0) {
        ring_layer_ = -1;
        pf_window_start();
    }
    ring_layer_ = layer;
    ring_waiting_ = false;
    dc_admit_list_.clear();
    int n = 0, njobs = 0;                  // n = blobs claimed, njobs = sub-reads published
    int64_t miss_bytes = 0;
    ++blob_stats_.calls;
    for (int64_t i = 0; i < k && n < kRingSlots; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_ || pf_seen_[(size_t) e]) continue;
        pf_seen_[(size_t) e] = 1;
        ++blob_stats_.entries;
        const int64_t idx = layer * n_expert_ + e;
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
            // R8: a resident expert must not keep a stale ring claim from an earlier window - with the
            // dynamic tier an expert CAN become resident between windows, and a stale `ring_of_` would both
            // misroute it to pass 1 and (once the slot is reused by another layer) serve the WRONG BYTES.
            ring_of_[(size_t) idx] = -1;
            ++blob_stats_.hot_skips;
            continue;
        }
        const uint64_t len = lay.blob_bytes(layer), off = lay.blob_offset(layer, e);
        if (len == 0 || off + len > file_map_bytes_) continue;
        const int slot = n++;
        // R8: **THE RING READ LANDS DIRECTLY IN AN LRU SLOT.**  A miss is admitted to the tier before the
        // read is issued, so the commit in `wait_layer` is one pointer publish - not a 2.2 MB copy per miss
        // per layer.  The LRU epoch is stamped at allocation, so a later claim in the SAME begin_layer call
        // can never evict an earlier one, and no victim can be a blob this window is still using.
        uint8_t* dst = ring_[(size_t) slot].data();
        if (dc_slots_ > 0) {
            const int32_t dc = dc_alloc(idx);
            if (dc >= 0) {
                dst = hot_arena_ + (uint64_t) dc * dc_slot_bytes_;
                dc_admit_list_.push_back(((int64_t) dc << 32) | (int64_t) (uint32_t) idx);
            } else {
                ++dc_fallbacks_;   // every slot touched this window: legacy ring, no admission
            }
        }
        // kSplit 1 must be exactly one read of the whole blob - `kChunk` only caps the SPLIT case.  Getting
        // this wrong left the engine issuing five 512 KiB reads per blob, which measured 8.5 tok/s against
        // 10.7 for the whole-blob form.
        const int ksplit = kSplit();
        const uint64_t step = ksplit <= 1 ? len
                              : std::min<uint64_t>(kChunk, (len + (uint64_t) ksplit - 1) / (uint64_t) ksplit);
        for (uint64_t at = 0; at < len && njobs < kRingSlots * kMaxChunks; at += step) {
            const uint64_t n = std::min<uint64_t>(step, len - at);
            pf_jobs_[(size_t) njobs++] = {dst + at, off + at, n};
        }
        ring_of_[(size_t) idx] = slot;
        pf_record_miss(idx);
        miss_bytes += (int64_t) len;
    }
    if (njobs == 0) return;
    // The rest: entries that are NOT resident and did NOT get a ring slot - i.e. a layer whose distinct
    // non-resident experts exceed `kRingSlots`.  They get one whole-blob WILLNEED each.
    //
    // **R7: THE HOT-TIER GUARD BELOW IS THE FIX FOR A 6.4x READ AMPLIFICATION.**  `pf_seen_[e]` is set to 1
    // for EVERY deduped id above, hot ones included, so this loop walked all of them; the only filter was
    // `ring_of_[idx] >= 0`, and a HOT expert never got a ring slot, so its test passed and the engine asked
    // the kernel to read it off the SSD into the page cache.  Measured on the IQ3_XXS pack, `--hot-ram-gib
    // 12`, one 150-token generation: 2,880 begin_layer calls, 67,748 ids seen, 52,741 hot-skipped, **96,231
    // blobs fetched** - against the 15,007 that were actually missed.  The 81,224 extra blobs were 176.8 GB
    // of readahead for experts already resident in RAM, and they were the whole of the decode time (209.4 GB
    // in 62.5 s = 3.35 GB/s, exactly the window time).  A resident expert must not be re-read from the drive,
    // whatever the reason.
    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_ || !pf_seen_[(size_t) e]) continue;
        pf_seen_[(size_t) e] = 2;       // mark as handled; ring_of_ decides who is actually served
        const int64_t idx = layer * n_expert_ + e;
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) continue;   // RESIDENT: nothing to read, ever
        if (ring_of_[(size_t) idx] >= 0) continue;
        const uint64_t len = lay.blob_bytes(layer), off = lay.blob_offset(layer, e);
        if (len == 0 || off + len > file_map_bytes_) continue;
        ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
        ++prefetched_;
        ++blob_stats_.disk;
        blob_stats_.disk_bytes += (int64_t) len;
        pf_record_miss(idx);
    }
    // publish the batch; the WAIT is now `wait_layer`, so the caller can put the hot tier's own expert work in
    // front of it.  The parked-count barrier (ExpertPool's protocol) guarantees every reader thread is parked
    // before head_/done_/njobs_ are reset, so no thread can touch the previous batch's state while it is
    // republished.  `pf_njobs_` is published together with `pf_pending_`, and `ring_pending` reads that flag,
    // so no entry can be observed as "still coming" after its bytes have landed.
    while (pf_parked_.load(std::memory_order_acquire) != (uint32_t) pf_workers_.size()) _mm_pause();
    pf_head_.store(0, std::memory_order_relaxed);
    pf_done_.store(0, std::memory_order_relaxed);
    pf_jobs_active_ = njobs;
    pf_njobs_.store((uint32_t) njobs, std::memory_order_release);
    pf_epoch_.fetch_add(1, std::memory_order_release);
    { std::lock_guard<std::mutex> lk(pf_mu_); pf_cv_.notify_all(); }
    blob_stats_.disk += n;
    blob_stats_.disk_bytes += miss_bytes;
    ring_waiting_ = true;
}

// R7: warm the page cache for a layer's whole expert set.  Prefill calls this once per layer, with the
// experts it is about to stage, before the first staging read.  See the header for why: prefill is QD1 and
// measured 0.67 GB/s on a 33k-token prompt, i.e. it was latency-bound exactly like decode was.
void ArenaExpertSource::prefetch(int64_t layer, const int32_t* experts, int64_t n) {
    if (file_fd_ < 0 || file_map_ == nullptr || experts == nullptr || n <= 0) return;
    if (layer < 0 || layer >= strata::kernels::cpu::expert_layout().n_layers) return;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0) return;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t e = experts[i];
        if (e < 0 || e >= n_expert_) continue;
        const int64_t idx = layer * n_expert_ + e;
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) continue;   // the tier answers it from RAM
        const uint64_t off = lay.blob_offset(layer, e);
        if (off + len > file_map_bytes_) continue;
        ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
        ++prefetched_;
    }
}

// R7: the blocking half of `begin_layer`.  Same wait the old single-phase form did inline.
// R7: a new verify window.  Score the previous window as a predictor of this one, then (optionally) ask the
// kernel to start reading the predicted set NOW, so the per-layer `pread` finds it in the page cache instead of
// paying the drive's QD1 latency 48 times.
//
// WHY THIS IS THE RIGHT LEVER.  Measured on this box's SNV2S1000G: one 2.08 MB random read costs **2.82 ms**
// (740 MB/s) at QD1, and the same read at QD2+ costs 0.06 ms once the page is resident.  The engine needs ~1-2
// such reads per layer and there are 48 layers, so the drive's LATENCY - not its bandwidth - sets the floor:
// 48 x 2.9 ms = 139 ms per window.  `POSIX_FADV_WILLNEED` on the predicted blobs moves that work to the start
// of the window where it can run at full queue depth beside the CPU, and the 106-odd syscalls cost microseconds.
void ArenaExpertSource::pf_window_start() {
    ++window_epoch_;
    // R8.1: decay the frequency counts.  Half-life = one window: a blob hit last window keeps most of its
    // credit, a blob untouched for eight windows has none, and the tier follows the workload as it moves.
    // ~10,840 shifts per window - noise.
    for (size_t i = 0; i < dc_count_.size(); ++i) dc_count_[i] >>= 1;
    // R8: the fadvise predictor is OFF by default now.  With the LRU tier, a blob that misses again is
    // ADMITTED (the ring reads it into a tier slot), so warming its page cache with WILLNEED only duplicates
    // the read - and on a cold page cache each call costs ~2 ms of synchronous kernel work, which at ~1,600
    // misses per adapting window was seconds per window of pure syscall.  STRATA_PREDICT=1 restores the R7
    // behaviour for A/B.
    if (window_epoch_ > 1 && prefetch_predict_ && file_fd_ >= 0) {
        for (const int32_t idx : last_misses_) {
            if (idx < 0 || idx >= blobs_) continue;
            const int64_t l = idx / n_expert_, e = idx % n_expert_;
            if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) continue;
            const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
            const uint64_t off = lay.blob_offset(l, e), len = lay.blob_bytes(l);
            if (len == 0 || off + len > file_map_bytes_) continue;
            ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
            ++blob_stats_.win_prefetched;
        }
    }
    last_misses_.swap(cur_misses_);
    cur_misses_.clear();
}

void ArenaExpertSource::pf_record_miss(int64_t idx) {
    if (idx < 0 || idx >= blobs_) return;
    if (miss_epoch_.size() != (size_t) blobs_) miss_epoch_.assign((size_t) blobs_, 0);
    if (miss_epoch_[(size_t) idx] == window_epoch_) return;              // already recorded this window
    if (miss_epoch_[(size_t) idx] == window_epoch_ - 1) ++blob_stats_.win_repeat;
    miss_epoch_[(size_t) idx] = window_epoch_;
    cur_misses_.push_back((int32_t) idx);
    ++blob_stats_.win_misses;
}

void ArenaExpertSource::wait_layer() {
    if (ring_waiting_) {
        const int n = pf_jobs_active_;
        while (pf_done_.load(std::memory_order_acquire) != (uint32_t) n) _mm_pause();
        prefetched_ += n;
        ring_waiting_ = false;
    }
    // R8: commit this layer's admissions.  The reads are confirmed landed, so publishing the slot offsets
    // now turns every one of them into a RAM hit for the rest of the session - and every `blob()` after
    // this point (the pass-1 dispatch) serves the tier copy, never the ring.
    if (!dc_admit_list_.empty()) {
        for (const int64_t v : dc_admit_list_) {
            const int32_t s = (int32_t) ((uint64_t) v >> 32);
            const int64_t idx = (int64_t) (int32_t) ((uint64_t) v & 0xffffffffu);
            if (s < 0 || s >= (int32_t) dc_slots_ || idx < 0 || idx >= (int64_t) hot_slot_.size()) continue;
            hot_slot_[(size_t) idx] = (int64_t) ((uint64_t) s * dc_slot_bytes_);
            ++dc_admits_;
        }
        dc_admit_list_.clear();
    }
}

// R7: is `(layer, expert)`'s blob one of the reads this layer submitted to the ring?
//
// **THIS MUST NOT TEST `ring_waiting_`.**  That flag means "wait_layer has not run yet", and it is cleared
// *between* dispatch pass 0 and pass 1 - so a version of this predicate that checked it made every pass-1
// entry look like a pass-0 entry, pass 1 skipped all of them, and **~19% of this model's expert
// contributions were silently dropped**: measured on the IQ3_XXS pack, 97 windows, 78,855 routed ids,
// 63,930 blob() calls - exactly the hot-tier count, i.e. not one miss was ever dispatched.  The model still
// produced fluent English, which is precisely why it had to be found in the counters rather than in the
// output; it is also the likely source of the capability eval's arithmetic misses.  A blob is "pending"
// when it was claimed for THIS layer, whatever stage of the two-pass handoff it is at.
bool ArenaExpertSource::ring_pending(int64_t layer, int64_t expert) const {
    if (layer != ring_layer_) return false;
    if (expert < 0 || expert >= n_expert_) return false;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_ || (int64_t) ring_of_.size() != blobs_) return false;
    return ring_of_[(size_t) idx] >= 0;
}

}  // namespace strata::core
