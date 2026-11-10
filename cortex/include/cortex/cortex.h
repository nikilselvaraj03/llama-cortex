// cortex — an OS-inspired memory-hierarchy governor for LLM inference runtimes.
//
// TL;DR
// -----
// The interesting part of serving an LLM is not the arithmetic, it is moving the
// right bytes to the right place at the right time.  Every serious inference
// engine has independently re-derived the classic operating-systems playbook for
// doing that: paging (PagedAttention), memory tiering (Pie, JetFire), readahead
// (SGLang HiCache prefetch), page replacement (HotPrefix, H2O), copy-on-write
// (vLLM prefix sharing), and granularity tuning (hugepages, neuron clusters).
//
// cortex makes that analogy first-class: it exposes the memory hierarchy of a
// machine as a *policy problem* and provides replaceable, testable
// implementations of each policy — so an inference runtime can pick, measure and
// swap them instead of hard-coding one.
//
// The policies here are deliberately chosen to sit in the gaps left by current
// engines (see docs/OS_TUNING.md):
//   * predictive, routing-driven readahead  (vs. reactive LRU)
//   * refault-distance-aware replacement     (vs. plain LRU / ClassLRU)
//   * copy-on-write sharing for *weights*    (engines only share KV)
//   * bandwidth-adaptive block granularity   (vs. a fixed block size)
//   * frequency-gated + pressure-aware admission
//   * measured cost-model placement planning (vs. assumed bandwidths)
//
// Everything in this header is deliberately free of runtime dependencies so it
// can be unit-tested and simulated deterministically, off-device.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortex {

// ===========================================================================
// 1. Memory hierarchy model
// ===========================================================================

enum class tier {
    vram         = 0, // device / unified-memory working set
    wired_ram    = 1, // host RAM that can be pinned (mlock / residency set)
    pageable_ram = 2, // ordinary anonymous + page-cache memory
    disk         = 3, // SSD / NVMe / object store
};

const char * tier_name(tier t);

struct tier_info {
    tier   t;
    size_t capacity_bytes;
    double bandwidth_gbps; // *effective* bandwidth, ideally measured
    double latency_us;     // fixed per-access / per-transfer latency
};

// A cost model over the memory hierarchy.  Costs are in microseconds and are
// intentionally simple: a fixed latency plus a serialization term at the
// slower of the two tiers' bandwidths.  That is the term that dominates when
// you move weights on every token.
class memory_hierarchy {
public:
    std::vector<tier_info> tiers;

    static memory_hierarchy make(size_t vram_bytes,
                                 size_t ram_bytes,
                                 size_t disk_bytes,
                                 double bw_vram_gbps,
                                 double bw_ram_gbps,
                                 double bw_disk_gbps);

    const tier_info & get(tier t) const;

    // cost to move `bytes` from src to dst (pays the slower end's latency)
    double transfer_cost_us(tier src, tier dst, size_t bytes) const;

    // cost to *read* `bytes` from tier t during one token step
    double residence_cost_us(tier t, size_t bytes) const;

    std::string describe() const;
};

// ===========================================================================
// 2. Adaptive transfer granularity
// ===========================================================================
//
// There is a fixed cost per transfer (kernel launch, DMA setup, page-table
// walk) and a per-byte cost (bandwidth).  Too-small blocks pay the fixed cost
// over and over; too-large blocks hurt reuse and increase tail latency.  The
// optimum is the smallest block whose transfer time amortises the fixed cost.
//
// This is the same trade-off as OS page size, and it is a knob almost nobody
// exposes (llama-modchip's hugepages, PowerInfer-2's neuron clusters and
// mzCache's chunks are the closest existing work).

struct granularity_model {
    double fixed_cost_us;     // per-transfer overhead
    double target_ratio;      // overhead must be <= 1/target_ratio of transfer time
    size_t min_bytes;
    size_t max_bytes;
};

size_t choose_block_bytes(const granularity_model & m, double bandwidth_gbps);

// ===========================================================================
// 3. Placement planner (measured cost model)
// ===========================================================================

struct placement_item {
    std::string name;
    size_t      bytes;
    double      access_freq;   // expected accesses per token (hotness)
    bool        must_resident; // e.g. pinned / required by the scheduler
};

struct placement_plan {
    std::vector<std::pair<std::string, tier>> assign;
    size_t resident_bytes   = 0;
    double est_cost_us      = 0.0; // modelled cost per token for this plan
    double baseline_cost_us = 0.0; // modelled cost per token if nothing is tiered
    bool   fits             = true;
    std::vector<std::string> explain;
};

// Greedy, hotness-density-first placement of weights + KV across the tiers.
// `kv_bytes` is placed first (it is touched on every token) and the remainder of
// the fastest tier is filled by value density (freq / bytes).
placement_plan plan_placement(const memory_hierarchy & h,
                              std::vector<placement_item> items,
                              size_t kv_bytes,
                              bool explain);

// ===========================================================================
// 4. Replacement policies (page replacement)
// ===========================================================================

enum class evict_policy {
    lru,      // baseline
    clock,    // second-chance, cheap approximation of LRU
    two_q,    // A1in/A1out/Am — protects against scan pollution
    s3fifo,   // small-FIFO + ghost + lazy promotion (2023)
    refault,  // MGLRU-inspired: promote blocks that refault quickly (anti-thrash)
};

const char * evict_policy_name(evict_policy p);

class eviction_index {
public:
    virtual ~eviction_index() = default;

    virtual const char * name() const = 0;

    // insert a key of `bytes`; evicts as needed. returns the keys evicted.
    virtual std::vector<uint64_t> insert(uint64_t key, size_t bytes) = 0;

    // record an access; returns true on a hit
    virtual bool touch(uint64_t key) = 0;

    virtual std::vector<uint64_t> evict_to_fit(size_t needed) = 0;

    virtual bool   contains(uint64_t key) const = 0;
    virtual size_t resident_bytes() const = 0;
    virtual size_t size() const = 0;
    virtual void   set_capacity(size_t bytes) = 0;
    virtual size_t capacity() const = 0;
};

std::unique_ptr<eviction_index> make_eviction(evict_policy p, size_t capacity_bytes);

// ===========================================================================
// 5. Trace replay / simulator
// ===========================================================================
//
// Deterministic, off-device replay of an access trace against any policy.  This
// is how the policies are compared without needing the same model twice.

struct sim_access {
    uint64_t key;
    size_t   bytes;
    uint64_t tick;
};

struct sim_result {
    uint64_t accesses      = 0;
    uint64_t hits          = 0;
    uint64_t misses        = 0;
    uint64_t evictions     = 0;
    double   hit_rate      = 0.0;
    double   bytes_moved   = 0.0;
    double   est_time_us   = 0.0;
    size_t   peak_resident = 0;
};

sim_result simulate(eviction_index & idx,
                    const std::vector<sim_access> & trace,
                    const memory_hierarchy & h,
                    tier from = tier::disk);

// ===========================================================================
// 6. Predictive readahead (routing-driven)
// ===========================================================================
//
// Existing engines prefetch reactively: a miss happens, then we fetch.  But MoE
// routing is strongly temporally correlated — the experts a layer will select
// are predictable from the experts the previous layer selected.  Observing
// routing lets us issue readahead for layer L+1 while layer L computes, which is
// exactly what a hardware prefetcher does for a strided access pattern.

class expert_prefetcher {
public:
    void observe(uint32_t layer, const std::vector<uint32_t> & experts);

    // predicted expert set for `layer`, given what layer-1 selected
    std::vector<uint32_t> predict(uint32_t layer, double min_confidence) const;

    // P(expert selected at `layer` | last observed set at layer-1), 0..1
    double confidence(uint32_t layer, uint32_t expert) const;

    size_t observations() const { return n_obs_; }

private:
    size_t n_obs_ = 0;
    std::unordered_map<uint32_t, std::vector<uint32_t>> last_;
    // trans_[from_layer][from_expert][to_expert] = count
    std::unordered_map<uint32_t,
        std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint64_t>>> trans_;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint64_t>> from_total_;
};

// ===========================================================================
// 7. Copy-on-write sharing registry
// ===========================================================================
//
// vLLM/SGLang share KV blocks across requests, but nobody shares *weights*.
// A registry of refcounted, content-addressed blocks lets several contexts (or
// several quantisations / LoRA variants of the same base) share one physical
// copy and release it only when the last reference goes away.

class cow_registry {
public:
    // intern a content-addressed block; refcount++. returns a stable handle.
    uint64_t acquire(uint64_t content_hash, size_t bytes);

    // drop one reference; returns true if the block became unreferenced
    bool release(uint64_t handle);

    size_t refcount(uint64_t handle) const;
    size_t physical_bytes() const;  // unique bytes actually stored
    size_t logical_bytes() const;   // sum of all acquisitions
    double dedup_ratio() const;     // logical / physical
    std::vector<uint64_t> evictable() const; // handles with refcount == 0

private:
    uint64_t next_handle_ = 1;
    size_t   physical_bytes_ = 0;
    size_t   logical_bytes_  = 0;
    struct block { uint64_t hash; size_t bytes; size_t refcount; };
    std::unordered_map<uint64_t, block> blocks_;        // handle -> block
    std::unordered_map<uint64_t, uint64_t> by_hash_;    // hash   -> handle
};

// ===========================================================================
// 8. Frequency-gated + pressure-aware admission
// ===========================================================================
//
// Promoting everything into the fast tier is what causes thrashing.  Gate
// promotion on an exponentially-decaying access frequency (double on hit, decay
// per step) and raise the bar as memory pressure rises — the same shape as
// TensorRT-LLM's KVBM SSD admission filter, generalised to any tier.

struct admission_config {
    double freq_threshold = 1.0; // base bar for promotion
    double decay_per_step = 0.5; // multiply all frequencies each step
    double pressure_high  = 0.85; // pressure at which the bar is raised
    double pressure_bar_scale = 4.0;
};

class admission_controller {
public:
    explicit admission_controller(admission_config cfg = {}) : cfg_(cfg) {}

    void   on_hit(uint64_t key);          // double this key's frequency
    void   on_step(double pressure);      // decay + record pressure
    bool   should_promote(uint64_t key) const;
    double frequency(uint64_t key) const;
    double pressure() const { return pressure_; }
    double effective_threshold() const;
    size_t admitted() const { return freq_.size(); }

private:
    admission_config cfg_;
    double pressure_ = 0.0;
    std::unordered_map<uint64_t, double> freq_;
};

// ===========================================================================
// 9. Write-back buffer for tier transitions
// ===========================================================================
//
// KV/weight updates that are headed for a slower tier should leave in batches,
// not one at a time — batching amortises the transfer's fixed cost and keeps the
// interconnect from being dominated by small messages.  (SGLang HiCache exposes
// write-through / write-back modes; this is the batching core of write-back.)

class writeback_buffer {
public:
    writeback_buffer(size_t max_dirty_bytes, size_t max_blocks)
        : max_dirty_bytes_(max_dirty_bytes), max_blocks_(max_blocks) {}

    void mark_dirty(uint64_t block, size_t bytes);
    bool should_flush() const;
    std::vector<uint64_t> flush(); // returns blocks to write, clears the buffer
    size_t dirty_bytes() const { return dirty_bytes_; }
    size_t dirty_blocks() const { return order_.size(); }
    size_t flushes() const { return flushes_; }

private:
    size_t max_dirty_bytes_;
    size_t max_blocks_;
    size_t dirty_bytes_ = 0;
    size_t flushes_    = 0;
    std::vector<uint64_t> order_;
    std::unordered_map<uint64_t, size_t> sizes_;
};

// ===========================================================================
// 10. Arena compaction / defragmentation
// ===========================================================================

struct arena_block {
    uint64_t id;
    size_t   begin;
    size_t   size;
    bool     alive;
};

struct compaction_result {
    size_t bytes_live          = 0;
    size_t bytes_total         = 0;
    size_t bytes_reclaimed     = 0;
    size_t moves               = 0; // live blocks that must be relocated
    double fragmentation_before = 0.0;
    std::vector<std::pair<uint64_t, size_t>> new_offsets; // id -> new begin
};

compaction_result compact_arena(const std::vector<arena_block> & blocks);

struct size_class_stats {
    size_t size;
    size_t count;
    size_t bytes;
};

// Size-class histogram: the justification for class-based allocation, which
// bounds fragmentation without moving hot blocks (the OS slab-allocator idea).
std::vector<size_class_stats> size_class_histogram(const std::vector<arena_block> & blocks);

} // namespace cortex
