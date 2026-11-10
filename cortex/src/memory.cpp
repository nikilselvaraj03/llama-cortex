// cortex — memory hierarchy model, granularity, placement planner, compaction.
#include "cortex/cortex.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <tuple>

namespace cortex {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::string fmt_bytes(size_t b) {
    char buf[64];
    if (b >= (size_t) (1ull << 30)) {
        snprintf(buf, sizeof buf, "%.2f GiB", double(b) / double(1ull << 30));
    } else {
        snprintf(buf, sizeof buf, "%.1f MiB", double(b) / double(1ull << 20));
    }
    return buf;
}

// ---------------------------------------------------------------------------
// 1. tier model
// ---------------------------------------------------------------------------

const char * tier_name(tier t) {
    switch (t) {
        case tier::vram:         return "vram";
        case tier::wired_ram:    return "wired_ram";
        case tier::pageable_ram: return "pageable_ram";
        case tier::disk:         return "disk";
    }
    return "unknown";
}

memory_hierarchy memory_hierarchy::make(size_t vram_bytes,
                                        size_t ram_bytes,
                                        size_t disk_bytes,
                                        double bw_vram_gbps,
                                        double bw_ram_gbps,
                                        double bw_disk_gbps) {
    memory_hierarchy h;

    // Host RAM is split into a *wired* tier (what we can realistically pin with
    // mlock / a Metal residency set) and a *pageable* tier.  The split matters
    // because a pinned page cannot be reclaimed under pressure -- which is the
    // whole point of pinning the hot set -- but you cannot pin everything.
    const size_t wired = ram_bytes / 4;

    h.tiers = {
        { tier::vram,         vram_bytes,                         bw_vram_gbps,      2.0   },
        { tier::wired_ram,    wired,                              bw_ram_gbps,       20.0  },
        { tier::pageable_ram, ram_bytes > wired ? ram_bytes-wired : 0, bw_ram_gbps * 0.6, 60.0 },
        { tier::disk,         disk_bytes,                         bw_disk_gbps,      500.0 },
    };
    return h;
}

const tier_info & memory_hierarchy::get(tier t) const {
    for (const auto & ti : tiers) {
        if (ti.t == t) {
            return ti;
        }
    }
    return tiers.back();
}

double memory_hierarchy::transfer_cost_us(tier src, tier dst, size_t bytes) const {
    const auto & s = get(src);
    const auto & d = get(dst);
    // A transfer is limited by the slower of the two links, and you pay at least
    // the larger of the two fixed latencies.
    const double bw = std::min(s.bandwidth_gbps, d.bandwidth_gbps);
    const double serialization_us = bw > 0.0 ? (double(bytes) / 1e9) / bw * 1e6 : 0.0;
    return std::max(s.latency_us, d.latency_us) + serialization_us;
}

double memory_hierarchy::residence_cost_us(tier t, size_t bytes) const {
    const auto & ti = get(t);
    const double serialization_us = ti.bandwidth_gbps > 0.0
        ? (double(bytes) / 1e9) / ti.bandwidth_gbps * 1e6
        : 0.0;
    return ti.latency_us + serialization_us;
}

std::string memory_hierarchy::describe() const {
    std::string s = "memory hierarchy (tier / capacity / bandwidth / latency):\n";
    for (const auto & t : tiers) {
        char buf[192];
        snprintf(buf, sizeof buf, "  %-13s %10s  %8.1f GB/s  %9.1f us\n",
                 tier_name(t.t), fmt_bytes(t.capacity_bytes).c_str(), t.bandwidth_gbps, t.latency_us);
        s += buf;
    }
    return s;
}

// ---------------------------------------------------------------------------
// 2. adaptive granularity
// ---------------------------------------------------------------------------
//
// Model:  t_transfer(B) = fixed_cost + B / bandwidth
// We want the smallest power-of-two block for which the fixed cost is at most
// 1/target_ratio of the transfer time, i.e. B >= fixed_cost * (ratio-1) * bw.
//
// Counter-intuitive but correct: a *faster* link needs a *larger* block to
// amortise the same fixed per-transfer cost, because each byte is cheaper
// there.  A slow link amortises the overhead for free.  Clamped to
// [min_bytes, max_bytes].  This is the OS page-size trade-off, applied across
// memory tiers instead of within RAM.

size_t choose_block_bytes(const granularity_model & m, double bandwidth_gbps) {
    if (bandwidth_gbps <= 0.0) {
        return m.min_bytes;
    }
    const double needed_bytes = (m.fixed_cost_us / 1e6) * m.target_ratio * bandwidth_gbps * 1e9;

    size_t b = m.min_bytes;
    while (b < m.max_bytes && (double) b < needed_bytes) {
        b <<= 1;
    }
    if (b < m.min_bytes) { b = m.min_bytes; }
    if (b > m.max_bytes) { b = m.max_bytes; }
    return b;
}

// ---------------------------------------------------------------------------
// 3. placement planner
// ---------------------------------------------------------------------------

placement_plan plan_placement(const memory_hierarchy & h,
                              std::vector<placement_item> items,
                              size_t kv_bytes,
                              bool explain) {
    placement_plan p;

    std::unordered_map<int, size_t> free_cap;
    for (const auto & t : h.tiers) {
        free_cap[(int) t.t] = t.capacity_bytes;
    }

    // -- 1. the KV cache is touched on every token, so it goes to the fastest
    //       tier that can hold it. -------------------------------------------
    tier kv_tier = tier::vram;
    bool kv_placed = false;
    for (tier t : { tier::vram, tier::wired_ram, tier::pageable_ram }) {
        if (free_cap[(int) t] >= kv_bytes) {
            kv_tier = t;
            free_cap[(int) t] -= kv_bytes;
            kv_placed = true;
            break;
        }
    }
    if (!kv_placed) {
        kv_tier = tier::disk;
        p.fits = false;
        if (free_cap[(int) tier::disk] >= kv_bytes) {
            free_cap[(int) tier::disk] -= kv_bytes;
        }
    }
    if (explain) {
        p.explain.push_back("kv_cache: " + fmt_bytes(kv_bytes) + " -> " + tier_name(kv_tier));
    }

    // -- 2. weights: fill the remaining fast capacity by value density
    //       (accesses per token per byte). ------------------------------------
    struct placed { std::string name; size_t bytes; double freq; tier t; };
    std::vector<placed> done;
    done.reserve(items.size());

    std::sort(items.begin(), items.end(),
              [](const placement_item & a, const placement_item & b) {
                  const double da = a.access_freq / double(std::max<size_t>(a.bytes, 1));
                  const double db = b.access_freq / double(std::max<size_t>(b.bytes, 1));
                  if (da != db) { return da > db; }
                  return a.name < b.name; // deterministic tie-break
              });

    for (const auto & it : items) {
        tier chosen = tier::disk;
        bool ok = false;
        for (tier t : { tier::vram, tier::wired_ram, tier::pageable_ram }) {
            if (free_cap[(int) t] >= it.bytes) {
                chosen = t;
                free_cap[(int) t] -= it.bytes;
                ok = true;
                break;
            }
        }
        if (!ok) {
            if (free_cap[(int) tier::disk] >= it.bytes) {
                free_cap[(int) tier::disk] -= it.bytes;
            } else {
                p.fits = false;
            }
            if (it.must_resident) {
                p.fits = false;
            }
        }
        done.push_back({ it.name, it.bytes, it.access_freq, chosen });
        if (chosen != tier::disk) {
            p.resident_bytes += it.bytes;
        }
        if (explain) {
            char buf[256];
            snprintf(buf, sizeof buf, "%-28s %10s  freq %6.2f/token -> %s",
                     it.name.c_str(), fmt_bytes(it.bytes).c_str(), it.access_freq, tier_name(chosen));
            p.explain.push_back(buf);
        }
    }

    // -- 3. cost model ------------------------------------------------------
    // per-token cost = sum_i freq_i * residence_cost(tier_i, bytes_i)  (+ kv)
    for (const auto & d : done) {
        p.est_cost_us      += d.freq * h.residence_cost_us(d.t, d.bytes);
        p.baseline_cost_us += d.freq * h.residence_cost_us(tier::disk, d.bytes);
        p.assign.emplace_back(d.name, d.t);
    }
    p.est_cost_us      += h.residence_cost_us(kv_tier, kv_bytes);
    p.baseline_cost_us += h.residence_cost_us(tier::disk, kv_bytes);

    if (explain) {
        char buf[192];
        snprintf(buf, sizeof buf,
                 "modelled per-token cost: %.1f us (tiered) vs %.1f us (all-on-disk) => %.2fx",
                 p.est_cost_us, p.baseline_cost_us,
                 p.est_cost_us > 0.0 ? p.baseline_cost_us / p.est_cost_us : 0.0);
        p.explain.push_back(buf);
    }
    return p;
}

// ---------------------------------------------------------------------------
// 10. compaction
// ---------------------------------------------------------------------------

compaction_result compact_arena(const std::vector<arena_block> & blocks) {
    compaction_result r;

    size_t total = 0;
    for (const auto & b : blocks) {
        total = std::max(total, b.begin + b.size);
    }

    std::vector<arena_block> alive;
    for (const auto & b : blocks) {
        if (b.alive && b.size > 0) {
            alive.push_back(b);
        }
    }
    std::sort(alive.begin(), alive.end(),
              [](const arena_block & a, const arena_block & b) { return a.begin < b.begin; });

    size_t cursor = 0;
    for (const auto & b : alive) {
        r.new_offsets.emplace_back(b.id, cursor);
        if (cursor != b.begin) {
            r.moves++; // this live block has to be relocated
        }
        cursor     += b.size;
        r.bytes_live += b.size;
    }

    r.bytes_total  = total;
    r.bytes_reclaimed = total > r.bytes_live ? total - r.bytes_live : 0;
    r.fragmentation_before = total > 0 ? double(total - r.bytes_live) / double(total) : 0.0;
    return r;
}

std::vector<size_class_stats> size_class_histogram(const std::vector<arena_block> & blocks) {
    std::unordered_map<size_t, size_class_stats> m;
    for (const auto & b : blocks) {
        if (!b.alive) {
            continue;
        }
        auto & s = m[b.size];
        s.size   = b.size;
        s.count += 1;
        s.bytes += b.size;
    }
    std::vector<size_class_stats> out;
    out.reserve(m.size());
    for (const auto & kv : m) {
        out.push_back(kv.second);
    }
    std::sort(out.begin(), out.end(),
              [](const size_class_stats & a, const size_class_stats & b) { return a.size < b.size; });
    return out;
}

} // namespace cortex
