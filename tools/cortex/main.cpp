// cortex — memory-hierarchy governor for an inference runtime.
//
// This tool answers, against the *actual* machine and the *actual* model:
//
//   1. What does the memory hierarchy look like?   (device VRAM, host RAM, disk,
//      and the bandwidth/latency cost model derived from it.)
//   2. Where should every weight and the KV cache live, and what does that
//      placement cost per token versus the naive placement?
//   3. Given the resulting working-set ratio, which replacement policy should
//      the engine install?  (Replayed off-device so the answer is reproducible.)
//
// It performs no inference and changes nothing: it is the *plan* step of the
// governor, the part you want to be able to read, diff and argue about.
//
// What-if overrides (env, so they cannot collide with common_params flags):
//   CORTEX_SIM_VRAM_MIB=<n>   pretend the device has n MiB
//   CORTEX_SIM_RAM_MIB=<n>    pretend the host has n MiB

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "cortex/cortex.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/statvfs.h>
#elif defined(__linux__)
#  include <unistd.h>
#  include <sys/statvfs.h>
#elif defined(_WIN32)
#  include <windows.h>
#endif

// ---------------------------------------------------------------------------
static std::string fmt_bytes(size_t b) {
    char buf[64];
    if (b >= (size_t) (1ull << 40)) {
        snprintf(buf, sizeof buf, "%.2f TiB", double(b) / double(1ull << 40));
    } else if (b >= (size_t) (1ull << 30)) {
        snprintf(buf, sizeof buf, "%.2f GiB", double(b) / double(1ull << 30));
    } else {
        snprintf(buf, sizeof buf, "%.1f MiB", double(b) / double(1ull << 20));
    }
    return buf;
}

static size_t env_mib(const char * name) {
    const char * v = getenv(name);
    if (!v || !*v) { return 0; }
    const long long mib = atoll(v);
    return mib > 0 ? (size_t) mib * (1u << 20) : 0;
}

// ---------------------------------------------------------------------------
static size_t host_ram_bytes() {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t   len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) { return (size_t) mem; }
#elif defined(__linux__)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long psize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psize > 0) { return (size_t) pages * (size_t) psize; }
#elif defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) { return (size_t) ms.ullTotalPhys; }
#endif
    return 0;
}

static size_t disk_free_bytes() {
#if defined(__APPLE__) || defined(__linux__)
    struct statvfs st;
    if (statvfs(".", &st) == 0) {
        return (size_t) st.f_bavail * (size_t) st.f_frsize;
    }
#endif
    return 0;
}

// ---------------------------------------------------------------------------
// 1. device enumeration
// ---------------------------------------------------------------------------
static void print_devices(size_t & vram_total, size_t & vram_free) {
    printf("== devices ==========================================================\n");

    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);

        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        const char * type_name = "?";
        switch (type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:  type_name = "cpu";  break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:  type_name = "gpu";  break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: type_name = "igpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL:type_name = "accel";break;
            case GGML_BACKEND_DEVICE_TYPE_META: type_name = "meta"; break;
        }

        printf("  [%zu] %-6s %-28s total %10s  free %10s\n",
               i, type_name,
               ggml_backend_dev_description(dev),
               fmt_bytes(total_b).c_str(), fmt_bytes(free_b).c_str());

        // the device that can hold the working set is the accelerator memory
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
            type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            vram_total += total_b;
            vram_free  += free_b;
        }
    }

    if (vram_total == 0) {
        printf("  (no dedicated accelerator memory; treating host RAM as the working set)\n");
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// 3. which replacement policy should the engine install?
// ---------------------------------------------------------------------------
static void print_policy_benchmark(size_t capacity_bytes, size_t hot_keys, const cortex::memory_hierarchy & h) {
    printf("== replacement policy (synthetic scan-pollution replay) ============\n");
    printf("  working set %s  |  %zu hot keys  |  one-shot scan larger than the cache\n",
           fmt_bytes(capacity_bytes).c_str(), hot_keys);

    // a small hot set, repeatedly reused, interleaved with a one-shot scan of
    // unique blocks. The scan exceeds the cache, so a recency-only policy drops
    // the entire hot set every round.
    // Size the blocks so the cache holds ~64 of them.  The scan is then 300
    // blocks, i.e. comfortably larger than the cache -- which is what makes the
    // scan actually pollute it.  (With 1 MiB blocks against a ~1 GiB cache the
    // scan would fit and every policy would tie, which tells you nothing.)
    const size_t block = capacity_bytes / 64 > 4096 ? capacity_bytes / 64 : 4096;

    std::vector<cortex::sim_access> trace;
    uint64_t t = 0, cold = 100000;
    for (int round = 0; round < 8; round++) {
        for (int i = 0; i < 64; i++) {
            trace.push_back({ (uint64_t)(i % (int) hot_keys), block, t++ });
        }
        for (int i = 0; i < 300; i++) {
            trace.push_back({ cold++, block, t++ });
        }
    }

    const cortex::evict_policy policies[] = {
        cortex::evict_policy::lru,  cortex::evict_policy::clock,
        cortex::evict_policy::two_q, cortex::evict_policy::s3fifo,
        cortex::evict_policy::refault,
    };

    const char * best = "lru";
    double best_hit  = -1.0;

    for (cortex::evict_policy p : policies) {
        auto idx = cortex::make_eviction(p, capacity_bytes);
        const auto r = cortex::simulate(*idx, trace, h, cortex::tier::disk);

        size_t hot = 0;
        for (uint64_t k = 0; k < hot_keys; k++) {
            if (idx->contains(k)) { hot++; }
        }
        printf("  %-8s hit_rate %6.2f%%   hot set retained %2zu/%zu   evictions %llu\n",
               cortex::evict_policy_name(p), r.hit_rate * 100.0, hot, hot_keys,
               (unsigned long long) r.evictions);

        if (r.hit_rate > best_hit) { best_hit = r.hit_rate; best = cortex::evict_policy_name(p); }
    }

    printf("  => recommended: --moe-cache-policy %s\n\n", best);
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    printf("cortex — memory-hierarchy governor\n\n");

    // ---- 1. hierarchy ------------------------------------------------------
    size_t vram_total = 0, vram_free = 0;
    print_devices(vram_total, vram_free);

    size_t ram = host_ram_bytes();
    size_t disk = disk_free_bytes();

    const size_t sim_vram = env_mib("CORTEX_SIM_VRAM_MIB");
    const size_t sim_ram  = env_mib("CORTEX_SIM_RAM_MIB");
    if (sim_vram) { vram_total = sim_vram; vram_free = sim_vram; printf("  [sim] vram overridden to %s\n", fmt_bytes(sim_vram).c_str()); }
    if (sim_ram)  { ram = sim_ram; printf("  [sim] ram overridden to %s\n", fmt_bytes(sim_ram).c_str()); }

    // measured-controller defaults for Apple-class unified memory; on a discrete
    // GPU the numbers are supplied by the device, so these are only the RAM/disk
    // tiers.  See docs/OS_TUNING.md for how to measure them properly.
    auto h = cortex::memory_hierarchy::make(
        vram_total, ram, disk,
        /*bw_vram=*/200.0, /*bw_ram=*/200.0, /*bw_disk=*/3.0);

    printf("== memory hierarchy =================================================\n");
    printf("%s\n", h.describe().c_str());

    // granularity: the transfer block size that amortises a fixed per-transfer
    // cost on each tier (see cortex::choose_block_bytes)
    const cortex::granularity_model gm{ /*fixed=*/5.0, /*ratio=*/16.0, 4096, 64ull << 20 };
    printf("  granularity (5 us fixed/transfer, 1/16 overhead budget):\n");
    for (const char * name : { "vram", "wired_ram", "pageable_ram", "disk" }) {
        cortex::tier t = cortex::tier::disk;
        if      (strcmp(name, "vram")         == 0) { t = cortex::tier::vram; }
        else if (strcmp(name, "wired_ram")    == 0) { t = cortex::tier::wired_ram; }
        else if (strcmp(name, "pageable_ram") == 0) { t = cortex::tier::pageable_ram; }
        printf("    %-13s %8zu KiB\n", name, cortex::choose_block_bytes(gm, h.get(t).bandwidth_gbps) >> 10);
    }
    printf("\n");

    // ---- 2. model geometry -------------------------------------------------
    printf("== model ============================================================\n");
    common_init_result init(params, /*model_only=*/true);
    llama_model * model = init.model();
    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 1;
    }

    char desc[256] = { 0 };
    llama_model_desc(model, desc, sizeof(desc));

    const uint64_t model_bytes = llama_model_size(model);
    const int32_t  n_layer     = llama_model_n_layer(model);
    const int32_t  n_embd      = llama_model_n_embd(model);
    const int32_t  n_head      = llama_model_n_head(model);
    const int32_t  n_head_kv   = llama_model_n_head_kv(model);
    const int32_t  n_ctx_train = llama_model_n_ctx_train(model);
    const int32_t  n_ctx       = params.n_ctx > 0 ? (int32_t) params.n_ctx : n_ctx_train;

    printf("  %s\n", desc);
    printf("  weights %s | %d layers | n_embd %d | heads %d/%d | ctx %d (train %d)\n",
           fmt_bytes(model_bytes).c_str(), n_layer, n_embd, n_head_kv, n_head, n_ctx, n_ctx_train);

    // KV cache: 2 (K and V) * n_embd_gqa per token per layer, fp16
    size_t kv_bytes = 0;
    if (n_layer > 0 && n_head > 0) {
        const int64_t n_embd_gqa = (int64_t) n_embd * n_head_kv / n_head;
        kv_bytes = (size_t) (2 * n_layer * (int64_t) n_ctx * n_embd_gqa * 2);
    }
    printf("  kv cache @ ctx %d: %s\n\n", n_ctx, fmt_bytes(kv_bytes).c_str());

    const bool is_moe = strstr(desc, "MoE") != nullptr ||
                        strstr(desc, "moe") != nullptr ||
                        strstr(desc, "A")   != nullptr; // e.g. "8x7B", "30B-A3B"
    if (is_moe) {
        printf("  note: MoE model — expert weights are used per-token with a lower\n"
               "        effective frequency than dense weights; the plan below treats\n"
               "        all layers as equally hot, which is a conservative upper bound\n"
               "        on how much VRAM they want.\n\n");
    }

    // ---- 3. placement plan -------------------------------------------------
    printf("== placement plan ===================================================\n");
    std::vector<cortex::placement_item> items;
    const size_t per_layer = n_layer > 0 ? (size_t) (model_bytes / (uint64_t) n_layer) : model_bytes;
    for (int32_t i = 0; i < (n_layer > 0 ? n_layer : 1); i++) {
        items.push_back({ "blk." + std::to_string(i), per_layer, 1.0, false });
    }

    const auto plan = cortex::plan_placement(h, items, kv_bytes, /*explain=*/true);
    for (const auto & line : plan.explain) {
        printf("    %s\n", line.c_str());
    }
    printf("\n    resident %s | fits: %s\n", fmt_bytes(plan.resident_bytes).c_str(), plan.fits ? "yes" : "NO");
    printf("\n");

    // ---- 4. policy ---------------------------------------------------------
    const size_t ws = plan.resident_bytes + kv_bytes;
    const size_t sim_cap = ws > 0 ? ws : (64ull << 20);
    const size_t hot = (size_t) (n_layer > 0 ? (n_layer < 64 ? n_layer : 64) : 16);
    print_policy_benchmark(sim_cap, hot, h);

    // ---- summary -----------------------------------------------------------
    printf("== summary ==========================================================\n");
    printf("  device working set : %s free of %s\n", fmt_bytes(vram_free).c_str(), fmt_bytes(vram_total).c_str());
    printf("  host ram           : %s\n", fmt_bytes(ram).c_str());
    printf("  weights            : %s\n", fmt_bytes(model_bytes).c_str());
    printf("  kv @ ctx %-8d : %s\n", n_ctx, fmt_bytes(kv_bytes).c_str());
    printf("  modelled cost      : %.1f us/token (tiered) vs %.1f us/token (all-on-disk)\n",
           plan.est_cost_us, plan.baseline_cost_us);
    printf("  recommended policy : see table above\n");

    llama_backend_free();
    return 0;
}
