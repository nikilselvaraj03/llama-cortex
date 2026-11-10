// cortex — unit tests. Deterministic, off-device: every policy is exercised by
// replaying a synthetic access trace so the behaviour is reproducible without a
// GPU or a model.
#include "cortex/cortex.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cortex;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                    \
            g_fail++;                                                                   \
        } else {                                                                        \
            g_pass++;                                                                   \
        }                                                                               \
    } while (0)

static const evict_policy k_policies[] = {
    evict_policy::lru,
    evict_policy::clock,
    evict_policy::two_q,
    evict_policy::s3fifo,
    evict_policy::refault,
};

// ---------------------------------------------------------------------------
static void test_tier_model() {
    printf("[ test ] memory hierarchy\n");
    auto h = memory_hierarchy::make(12ull << 30, 16ull << 30, 500ull << 30, 200.0, 200.0, 3.0);

    CHECK(h.get(tier::vram).capacity_bytes == (12ull << 30));

    const double c_vram = h.residence_cost_us(tier::vram, 1ull << 30);
    const double c_disk = h.residence_cost_us(tier::disk, 1ull << 30);
    CHECK(c_vram < c_disk);

    // bringing a GiB up from disk must cost more than from host RAM
    CHECK(h.transfer_cost_us(tier::disk, tier::vram, 1ull << 30) >
          h.transfer_cost_us(tier::wired_ram, tier::vram, 1ull << 30));
    printf("  ok (%.1f us vs %.1f us to read 1 GiB)\n", c_vram, c_disk);
}

// ---------------------------------------------------------------------------
static void test_granularity() {
    printf("[ test ] adaptive granularity\n");
    granularity_model m{ 5.0, 16.0, 4096, 64ull << 20 }; // 5 us fixed per-transfer cost

    const size_t fast = choose_block_bytes(m, 200.0);  // unified memory
    const size_t slow = choose_block_bytes(m, 3.0);    // NVMe

    CHECK((fast & (fast - 1)) == 0); // power of two
    CHECK(fast >= 4096 && fast <= (64ull << 20));
    CHECK((slow & (slow - 1)) == 0);

    // a *faster* link needs the *larger* block: bytes are cheaper there, so
    // more of them are needed to amortise the same fixed cost.
    CHECK(fast >= slow);
    printf("  ok (200 GB/s -> %zu KiB, 3 GB/s -> %zu KiB)\n", fast >> 10, slow >> 10);
}

// ---------------------------------------------------------------------------
static void test_planner() {
    printf("[ test ] placement planner\n");
    auto h = memory_hierarchy::make(1ull << 30, 16ull << 30, 100ull << 30, 200.0, 200.0, 3.0);

    std::vector<placement_item> items = {
        { "token_embd", 200ull << 20, 1.00, false },
        { "blk.0-31",   300ull << 20, 1.00, false },
        { "blk.32-63",  300ull << 20, 0.25, false },
    };
    auto p = plan_placement(h, items, 300ull << 20, /*explain=*/true);

    CHECK(p.fits);
    CHECK(p.assign.size() == 3);
    CHECK(p.est_cost_us > 0.0);
    CHECK(p.baseline_cost_us > p.est_cost_us); // tiering must beat all-on-disk
    CHECK(!p.explain.empty());

    for (const auto & line : p.explain) {
        printf("    %s\n", line.c_str());
    }
    printf("  ok (%.1f us/token tiered vs %.1f us all-on-disk)\n",
           p.est_cost_us, p.baseline_cost_us);
}

// ---------------------------------------------------------------------------
static void test_eviction_invariants() {
    printf("[ test ] eviction invariants\n");

    for (evict_policy pol : k_policies) {
        auto idx = make_eviction(pol, 1u << 20); // 256 x 4 KiB

        for (uint64_t k = 0; k < 1000; k++) {
            idx->insert(k, 4096);
        }
        CHECK(idx->resident_bytes() <= idx->capacity());
        CHECK(idx->size() > 0);

        // touching everything resident must not grow the footprint
        const size_t before = idx->resident_bytes();
        for (uint64_t k = 0; k < 1000; k++) {
            idx->touch(k);
        }
        CHECK(idx->resident_bytes() == before);
        CHECK(idx->resident_bytes() <= idx->capacity());

        printf("  ok %-8s size=%-5zu resident=%6.0f KiB (cap %6.0f KiB)\n",
               evict_policy_name(pol), idx->size(),
               double(idx->resident_bytes()) / 1024.0, double(idx->capacity()) / 1024.0);
    }
}

// ---------------------------------------------------------------------------
// Scan pollution: a small hot set (keys 0..15) is accessed repeatedly, then a
// long one-shot scan of unique keys (never reused) blows through the cache.
// This is *the* pattern that defeats LRU, and the reason 2Q / S3-FIFO / MGLRU
// style policies exist.  The scan is larger than the cache, so plain LRU
// evicts the entire hot set every round.
static std::vector<sim_access> make_scan_pollution_trace() {
    std::vector<sim_access> trace;
    uint64_t t    = 0;
    uint64_t cold = 1000;

    for (int round = 0; round < 8; round++) {
        for (int i = 0; i < 64; i++) {
            trace.push_back({ (uint64_t)(i % 16), 1ull << 20, t++ }); // hot
        }
        for (int i = 0; i < 300; i++) {
            trace.push_back({ cold++, 1ull << 20, t++ });             // one-shot scan
        }
    }
    return trace;
}

static void test_simulator_compares_policies() {
    printf("[ test ] replay simulator (scan-pollution trace)\n");

    auto h = memory_hierarchy::make(8ull << 30, 32ull << 30, 0, 200.0, 200.0, 3.0);
    const auto trace = make_scan_pollution_trace();

    double lru_hit   = -1.0;
    double best_hit  = -1.0;
    size_t lru_hot   = 0;
    size_t best_hot  = 0;
    std::string best_name;

    for (evict_policy pol : k_policies) {
        auto idx = make_eviction(pol, 64ull << 20); // 64 MiB = 64 x 1 MiB blocks
        const auto r = simulate(*idx, trace, h, tier::disk);

        size_t hot = 0;
        for (uint64_t k = 0; k < 16; k++) {
            if (idx->contains(k)) { hot++; }
        }

        printf("  %-8s hit_rate=%.4f  hot_resident=%2zu/16  bytes_moved=%7.1f GiB  evictions=%llu\n",
               evict_policy_name(pol), r.hit_rate, hot,
               r.bytes_moved / double(1ull << 30), (unsigned long long) r.evictions);

        CHECK(r.accesses == trace.size());
        CHECK(r.hits + r.misses == r.accesses);
        CHECK(r.peak_resident <= (64ull << 20));
        CHECK(idx->resident_bytes() <= idx->capacity());

        if (pol == evict_policy::lru) { lru_hit = r.hit_rate; lru_hot = hot; }
        if (r.hit_rate > best_hit)    { best_hit = r.hit_rate; best_hot = hot; best_name = evict_policy_name(pol); }
    }

    // The claim being tested: under scan pollution at least one of the
    // non-LRU policies is strictly better than LRU, and protects the hot set.
    CHECK(best_hit > lru_hit);
    CHECK(best_hot > lru_hot);
    CHECK(lru_hot == 0); // LRU loses the whole hot set to the scan

    printf("  ok (best: %s hit %.4f vs lru %.4f; hot retained %zu vs %zu)\n",
           best_name.c_str(), best_hit, lru_hit, best_hot, lru_hot);
}

// ---------------------------------------------------------------------------
static void test_prefetcher() {
    printf("[ test ] predictive prefetcher\n");

    expert_prefetcher pf;

    // planted structure: at layer 1, expert 7 always follows expert 3;
    // expert 9 is noise that appears 10% of the time.
    for (int step = 0; step < 200; step++) {
        pf.observe(0, { 3 });
        pf.observe(1, (step % 10 == 0) ? std::vector<uint32_t>{ 7, 9 }
                                       : std::vector<uint32_t>{ 7 });
    }

    const auto pred = pf.predict(1, 0.8);
    CHECK(!pred.empty());
    CHECK(pred[0] == 7);              // the predictable one comes first
    CHECK(pf.confidence(1, 7) > 0.8); // P(7 | 3) ~= 0.9
    CHECK(pf.confidence(1, 9) < 0.8); // noise must NOT clear the confidence bar
    printf("  ok (confidence(7)=%.2f, confidence(9)=%.2f, predicted=%zu)\n",
           pf.confidence(1, 7), pf.confidence(1, 9), pred.size());
}

// ---------------------------------------------------------------------------
static void test_cow_registry() {
    printf("[ test ] copy-on-write registry\n");

    cow_registry reg;
    const uint64_t a = reg.acquire(0xAAAA, 100ull << 20);
    const uint64_t b = reg.acquire(0xAAAA, 100ull << 20); // same content, shared
    const uint64_t c = reg.acquire(0xBBBB, 50ull << 20);

    CHECK(a == b);
    CHECK(reg.refcount(a) == 2);
    CHECK(reg.physical_bytes() == (150ull << 20));
    CHECK(reg.logical_bytes()  == (250ull << 20));

    const double ratio = reg.dedup_ratio(); // capture while both refs are live
    CHECK(ratio > 1.6);

    CHECK(!reg.release(a));          // still referenced by b
    CHECK(reg.refcount(a) == 1);
    CHECK(reg.release(b));           // last reference -> block freed
    CHECK(reg.refcount(a) == 0);
    CHECK(reg.evictable().empty());  // c is still referenced

    CHECK(reg.release(c));           // free the last block
    CHECK(reg.physical_bytes() == 0);
    printf("  ok (dedup ratio %.2fx, physical now %.0f MiB)\n",
           ratio, double(reg.physical_bytes()) / double(1ull << 20));
}

// ---------------------------------------------------------------------------
static void test_admission() {
    printf("[ test ] pressure-aware admission\n");

    admission_controller ac;

    ac.on_hit(1000);
    CHECK(ac.frequency(1000) >= 1.0);

    ac.on_step(0.10); // low pressure
    const double low_bar = ac.effective_threshold();

    ac.on_step(0.98); // high pressure
    const double high_bar = ac.effective_threshold();

    CHECK(high_bar > low_bar); // pressure raises the promotion bar
    printf("  ok (bar %.2f -> %.2f as pressure rises)\n", low_bar, high_bar);
}

// ---------------------------------------------------------------------------
static void test_writeback_and_compaction() {
    printf("[ test ] write-back + compaction\n");

    writeback_buffer wb(4ull << 20, 1000);
    CHECK(!wb.should_flush());
    for (uint64_t b = 0; b < 10; b++) { wb.mark_dirty(b, 1ull << 20); }
    CHECK(wb.should_flush());
    const auto batch = wb.flush();
    CHECK(batch.size() == 10);
    CHECK(wb.dirty_bytes() == 0);
    CHECK(wb.flushes() == 1);
    wb.mark_dirty(1, 1ull << 20); // re-dirty after flush
    CHECK(wb.dirty_bytes() == (1ull << 20));

    // arena: 3 live blocks separated by 2 dead holes
    std::vector<arena_block> arena = {
        { 1, 0,          1ull << 20, true  },
        { 2, 1ull << 20, 1ull << 20, false }, // hole
        { 3, 2ull << 20, 1ull << 20, true  },
        { 4, 3ull << 20, 1ull << 20, false }, // hole
        { 5, 4ull << 20, 1ull << 20, true  },
    };
    const auto cr = compact_arena(arena);
    CHECK(cr.bytes_live == (3ull << 20));
    CHECK(cr.bytes_total == (5ull << 20));
    CHECK(cr.bytes_reclaimed == (2ull << 20));
    CHECK(cr.moves >= 1);
    CHECK(cr.fragmentation_before > 0.39 && cr.fragmentation_before < 0.41);

    const auto hist = size_class_histogram(arena);
    CHECK(hist.size() == 1 && hist[0].count == 3);
    printf("  ok (reclaimed %.0f MiB, %zu moves, frag %.0f%%)\n",
           double(cr.bytes_reclaimed) / double(1ull << 20), cr.moves,
           cr.fragmentation_before * 100.0);
}

// ---------------------------------------------------------------------------
int main(void) {
    printf("cortex unit tests\n=================\n");

    test_tier_model();
    test_granularity();
    test_planner();
    test_eviction_invariants();
    test_simulator_compares_policies();
    test_prefetcher();
    test_cow_registry();
    test_admission();
    test_writeback_and_compaction();

    printf("=================\n%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
