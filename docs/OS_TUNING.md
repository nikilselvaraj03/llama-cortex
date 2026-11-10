# cortex — an OS-inspired memory-hierarchy governor for LLM inference

> **Thesis.** The hard part of serving an LLM is not the arithmetic, it is
> moving the right bytes to the right place at the right time. Every serious
> inference engine has independently re-derived the operating system's playbook
> for doing that — paging, tiering, readahead, page replacement,
> copy-on-write, and granularity tuning. `cortex` makes that analogy
> first-class: it exposes a machine's memory hierarchy as a **policy problem**
> and ships replaceable, testable implementations of each policy.

`cortex` is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). It adds
a standalone governor library (`cortex/`) and a planning tool
(`llama-cortex`) on top of the existing engine.

---

## 1. Why this analogy is not just a metaphor

The correspondence is structural, not decorative. Table stakes for the OS and
the inference runtime are the same problem statement: *a working set larger than
the fastest memory, accessed with reuse and skew, under a fixed capacity.*

| Operating system | LLM inference analogue | Where it shows up |
|---|---|---|
| Virtual memory + page tables | Paged KV cache with a block table | vLLM **PagedAttention** (SOSP'23) |
| Page replacement (LRU/CLOCK/2Q/ARC) | KV block eviction, expert eviction | **HotPrefix**; llama.cpp MoE `ClassLRU` |
| Demand paging | Expert/layer offload on first use | KTransformers, Mixtral-offloading |
| Readahead / hardware prefetch | Speculative weight/next-layer prefetch | SGLang **HiCache** |
| Copy-on-write + dedup | Prefix sharing between requests | vLLM / SGLang radix sharing |
| Memory tiering (DRAM/NVMe) | VRAM → pinned host → pageable → disk | **Pie**; **JetFire** |
| Page size / TLB granularity | Transfer block size, tile size, quant group | llama-modchip hugepages; PowerInfer-2 clusters |
| Admission control / cgroups | Frequency-gated promotion of hot blocks | TensorRT-LLM **KVBM** SSD admission |
| Compaction / defragmentation | KV arena defrag | llama.cpp KV defrag |
| Write-back vs write-through | Batched KV flush to a slower tier | SGLang HiCache write policies |

The distinction `cortex` draws is **which cells are still empty**. As of late
2026 the crowd is squarely on *KV* paging and prefix reuse. The comparatively
thin areas — and therefore where `cortex` aims — are:

1. **weight** paging with real fault-and-admission semantics,
2. **copy-on-write sharing for weights** (engines share KV, not weights),
3. **granularity as a tunable** rather than a constant,
4. **readahead driven by routing prediction** (llama.cpp's expert cache is
   reactive, not predictive),
5. **measured** cost models rather than assumed bandwidths.

---

## 2. The mechanisms

`cortex` ships fourteen distinct, individually testable mechanisms. Every one is
implemented in `cortex/src/` and exercised by `cortex/tests/test-cortex.cpp`.

### Memory model

**1. Tier model + cost model** (`memory_hierarchy`)
Four tiers — `vram`, `wired_ram`, `pageable_ram`, `disk` — each with capacity,
effective bandwidth and fixed latency. `transfer_cost_us(src, dst, bytes)` pays
the slower link's bandwidth and the larger tier's latency, which is the term
that dominates when weights move on every token.

**2. Adaptive granularity** (`choose_block_bytes`)
There is a fixed cost per transfer (kernel launch, DMA setup, page walk) and a
per-byte cost. The optimum block size is the smallest one whose transfer time
amortises the fixed cost. Note the counter-intuitive consequence, which the code
and its test both assert: **a faster link needs a *larger* block**, because each
byte is cheaper there, so more bytes are needed to accumulate the same transfer
time. A slow link amortises the overhead for free.

**3. Cost-model placement planner** (`plan_placement`)
Greedy, hotness-density-first (`freq / bytes`) placement of weights and the KV
cache across the tiers, with a modelled per-token cost for the chosen plan and
for the all-on-disk baseline, plus a human-readable explanation of every
decision. Unlike engines that hard-code "KV in VRAM, experts in RAM", the
placement is *computed from the machine you are actually on*.

### Replacement policies

**4–8. Five interchangeable eviction policies** (`make_eviction`)

| policy | idea | weakness it fixes |
|---|---|---|
| `lru` | recency only | baseline |
| `clock` | second-chance reference bit, O(1) | cost of LRU bookkeeping |
| `2q` | A1in / A1out ghost / Am protected LRU | scan pollution |
| `s3fifo` | small FIFO + ghost + lazy promotion | scan pollution, cheaper than 2Q |
| `refault` | MGLRU-style: two generations + reuse-distance window; a block that refaults *quickly* is thrashing and gets protected | **thrashing** |

`refault` is the interesting one. Plain LRU cannot distinguish "cold, evict it"
from "hot, but my working set is larger than the cache". A refault-distance
window can: if a block is evicted and demanded again within `window` operations,
it is by definition in the active working set, so it is promoted into the
protected generation and survives the next sweep of cold blocks.

**9. Trace replay simulator** (`simulate`)
Deterministic, off-device replay of an access trace against any policy. This is
what makes the policies *comparable* without running a model twice — and it is
how the numbers in §3 were produced.

### Runtime policies

**10. Predictive, routing-driven readahead** (`expert_prefetcher`)
Observes per-layer expert selection and learns the conditional distribution
`P(expert@layer | expert@layer-1)`. It then predicts layer *L+1*'s expert set
while layer *L* computes, gated by a confidence threshold so that speculative
fetching cannot waste bandwidth on unpredicted experts. This is a hardware
prefetcher for a learned, non-strided access pattern. Existing engines prefetch
*reactively* — on a miss.

**11. Copy-on-write registry** (`cow_registry`)
Refcounted, content-addressed blocks. Two contexts (or two quantisations, or two
LoRA variants of one base) can share a single physical copy; the block is
released only when the last reference goes away. Engines do this for KV blocks;
nobody does it for weights.

**12. Frequency-gated, pressure-aware admission** (`admission_controller`)
Promotion into the fast tier is gated on an exponentially-decaying access
frequency (double on hit, decay per step), and the bar *rises* with memory
pressure. Same shape as TensorRT-LLM's KVBM disk-admission filter, generalised
to any tier.

**13. Write-back batching** (`writeback_buffer`)
Updates headed for a slower tier leave in batches rather than one at a time,
amortising the fixed per-transfer cost and keeping the interconnect from being
dominated by small messages.

**14. Arena compaction + size-class histogram** (`compact_arena`,
`size_class_histogram`)
Computes the relocation set and reclaimed bytes for defragmenting an arena, and
reports the size-class distribution that justifies class-based allocation
(the OS slab-allocator idea) without ever moving hot blocks.

---

## 3. Measured results

Everything below is real output from this repository, on the machine in §5.
No numbers are estimated.

### Unit tests — `./build/bin/test-cortex`

```
85 checks passed, 0 failed
```

### Replacement policies under scan pollution

Trace: a 16-block hot set accessed repeatedly, interleaved with an 8-round
one-shot scan of unique blocks through a 64-block cache. This is the pattern
that defeats recency-only replacement.

| policy | hit rate | hot set retained | evictions |
|---|---|---|---|
| `lru` | 13.19 % | 0 / 16 | 2464 |
| `clock` | 13.19 % | 0 / 16 | 2464 |
| `2q` | 16.48 % | 16 / 16 | 2400 |
| `s3fifo` | 16.69 % | 16 / 16 | 2404 |
| **`refault`** | **17.03 %** | **16 / 16** | **2352** |

`refault` is **+29 % relative hit rate** over LRU and, unlike LRU/CLOCK, never
loses the hot set. The absolute margins are modest — which is the honest
picture, and precisely why replacement policy should be a *pluggable, measurable*
choice rather than a hard-coded constant.

### Planner

For a 1.34 GiB working set (1 GiB weights + 300 MiB KV) on a 1 GiB-VRAM
machine, the governor places the KV cache and hottest weights in VRAM and the
cold half of the model in wired RAM, against a modelled per-token cost of
**4 598 µs versus 307 460 µs** for the all-on-disk baseline (**66.9×**).

### Engine baseline

The unmodified engine on a 1B Q4_K_M model (Apple M1 Pro, Metal):

| test | throughput |
|---|---|
| `pp128` | 1279.06 ± 23.98 t/s |
| `tg64` | 118.12 ± 0.45 t/s |

Reproduce with `llama-bench -m <model.gguf> -p 128 -n 64 -r 2`.

---

## 4. Using it

```bash
# the governor: measure the machine, plan the placement, pick a policy
./build/bin/llama-cortex -m model.gguf -c 8192

# what-if: pretend the device has 2 GiB
CORTEX_SIM_VRAM_MIB=2048 ./build/bin/llama-cortex -m model.gguf -c 8192

# the policy library alone
./build/bin/test-cortex
```

`llama-cortex` reads the **real** device memory through ggml
(`ggml_backend_dev_memory`) and the **real** model geometry
(`llama_model_n_layer`, `llama_model_n_head_kv`, …), so its plan is specific to
your machine rather than to a datasheet.

---

## 5. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build -j --target llama-cortex test-cortex
```

`cortex` is a dependency-free C++17 static library; it is unit-testable and
simulatable with no GPU and no model present.

### Tested configuration

Apple M1 Pro (MacBookPro18,3), 8 cores, 16 GiB unified memory,
macOS 26.6.2, Apple clang 21, Metal backend, `recommendedMaxWorkingSetSize`
12 713 MiB.

> **Toolchain note (macOS).** If the link step fails with
> `tapi error: malformed file ... unknown architecture arm64e.x1-macos`, the
> active SDK is newer than the linker understands. Point at a matching SDK
> without needing sudo:
> `export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.

---

## 6. Prior art, and what is actually new

Honesty matters more than novelty, so: **most of these mechanisms have prior
art, and `cortex` does not claim to have invented them.**

| mechanism | prior art | what `cortex` adds |
|---|---|---|
| routing-predicted expert prefetch | **Mixtral-offloading** (`arXiv:2312.17238`) predicts the next layer's routing from the hidden state and speculatively prefetches experts | a confidence-gated conditional model `P(e@L \| e@L-1)` that is testable off-device |
| admission-controlled replacement | **HotPrefix** (SIGMOD'26) admits only hot prefixes into the CPU tier | generalised to any tier, with a pressure-scaled bar |
| prefetch-on-schedule | **Pie** (`arXiv:2411.09317`) hides paging because the layer order is known a priori | a cost model that *chooses* the schedule from measured bandwidth |
| MGLRU-style refault | Linux 6.1+ multi-generational LRU | applied to KV/expert blocks, with a replay simulator |
| COW weight sharing | vLLM / SGLang share **KV** blocks | the same idea applied to **weights** |

The contribution is the *packaging*: one dependency-free library where each
policy is a named, swappable, unit-tested object, plus a planner that measures
the machine instead of assuming a datasheet.

## 7. Apple Silicon caveat — read this before trusting the numbers

On Apple Silicon there is **one physical memory pool**. `vram` and `ram` are not
additive; the GPU working set is a *residency budget* carved out of the same
RAM. `llama-cortex` detects this (iGPU, or accelerator total > ½ host RAM) and
builds the hierarchy as `vram = working set`, `wired_ram = 0`,
`pageable_ram = ram − vram`, so the tiers do not double-count.

This also means the classic tiering playbook does not transfer directly.
*Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon*
(`arXiv:2610.31479`) finds that KV pruning can help on Apple while
**speculative decoding reduces throughput**, and that public Metal lacks
asynchronous copy — so batch-1 decode cannot hide a transfer. Treat the policy
table as a *decision aid for the working-set ratio*, not as a promise of
speedup on this hardware.

## 8. Integration map — where a governor actually plugs in

Anchored against commit `45c8ea8`, so the next step is a diff, not a search.

**KV / memory — the primary insertion point**
- `src/llama-memory.h:104` — `struct llama_memory_i`, the single interface every
  KV variant implements (`iswa`, `dsa`, `dsv4`, `msa`, `hybrid`, `recurrent`).
- `src/llama-model.cpp:2914` — `llama_model::create_memory()` picks the concrete
  cache; `src/llama-context.cpp:550` installs it.
- A **decorator** over `llama_memory_i` needs no changes to any variant:
  `init_batch` / `init_update` / `apply` all funnel through it, and
  `memory_breakdown()` (`src/llama-memory.h:151`) already reports bytes per
  `ggml_backend_buffer_type_t`, so VRAM/RAM accounting needs no new plumbing.

**Weights — placement**
- `src/llama-model.cpp:1677` `load_tensors()` — where `n_gpu_layers` becomes
  `i_gpu_start` and each layer is bound to a device; `:1742` already calls
  `ggml_backend_dev_memory()` to split by free VRAM.
- `src/llama-model.cpp:1340` `llama_model::impl` — the `cpu_buft_list` /
  `gpu_buft_list` that decide placement.

**Existing hooks a governor can use today**

| hook | anchor | granularity |
|---|---|---|
| `ggml_backend_sched_eval_callback` | `ggml/include/ggml-backend.h:333` | per graph node |
| `ggml_backend_sched_copy_callback` | `ggml-backend.h:342`, set at `src/llama-context.cpp:790` | per split weight copy |
| `ggml_backend_sched_prefetch_inputs` | `ggml/src/ggml-backend.cpp:2152` | per split, async prefetch |
| `llama_prefetch(ranges)` | `src/llama-mmap.h:90` | arbitrary host page ranges |
| `layer_filter_cb` / `layer_share_cb` | `src/llama-memory.h:106,112` | per layer, at cache construction |

**What llama.cpp already has, so `cortex` does not duplicate it**
- KV cache: a **flat, contiguous per-layer tensor** `[n_embd_k_gqa, kv_size,
  n_stream]` (`src/llama-kv-cache.cpp:233`) with *cell metadata* over it
  (`llama_kv_cells`, `src/llama-kv-cells.h:68`) and a ring/slot scan
  (`find_slot`, `src/llama-kv-cache.cpp:950`). There is **no block table and no
  eviction to host** — it is not paging.
- Prefix caching: **yes, but only at the server layer** — whole serialized state
  blobs in `server_prompt_cache` (`tools/server/server-task.h:733`), not
  KV-block granular.
- COW: **no.** Only aliasing via `layer_share_cb` / `mem_other`.
- The closest existing pager is `llama_moe_cache` (`src/llama-moe-cache.h:13`),
  an LRU/CRF device-resident expert cache — the template a governor should
  generalise.

## 9. Status — what is wired, and what is next

Stated plainly, because the difference matters.

**Wired and verified**
- the full policy library, with `-Wall -Wextra` clean and 85/85 tests green;
- the governor tool, reading real device memory and real model geometry;
- build integration into the CMake tree.

**Designed, documented, not yet wired into the decode hot path**
- installing a `refault`/`2q`/`s3fifo` replacement into the in-tree MoE expert
  cache (`src/llama-moe-cache.cpp`, currently `ClassLRU`/CRF) behind a
  `--moe-cache-policy` flag;
- driving `expert_prefetcher` from `llama-graph.cpp`'s `build_moe_cache_slots`;
- cross-context weight sharing through `cow_registry`.

These are deliberate scope boundaries, not omissions: each touches the decode
loop, and the governor's job is to make the *policy* explicit and measurable
before it is wired in.

---

## 10. References

Operating systems
- Corbet, Rubini, Kroah-Hartman — *Linux Device Drivers* / kernel MM docs
- **MGLRU** — multi-generational LRU, Linux 6.1+ (`refault`'s model)
- Yang et al. — *FIFO queues are all you need for cache eviction* (S3-FIFO, SOSP'23)
- Johnson & Shasha — *2Q: a low overhead high performance buffer management
  replacement approach* (VLDB'94)

Inference systems
- Kwon et al. — *Efficient Memory Management for LLM Serving with PagedAttention*
  (vLLM, SOSP'23) — `arXiv:2309.06180`
- Xu et al. — *Pie: Pooling CPU Memory for LLM Inference* — `arXiv:2411.09317`
- *Improving the Performance of Out-of-Core LLM Inference Using Heterogeneous
  Host Memory* (IISWC 2025)
- *Accelerating Model Loading in LLM Inference by Programmable Page Cache*
  (USENIX FAST'26)
- *Towards Segmentation-Based Address Translation for LLM Inference*
  (IEEE CAL 2026)
- Liu et al. — *Fast Vector Quantization / H2O, StreamingLLM, SnapKV* — KV
  eviction as page replacement
- *HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing*
  (ACM 2025)
- Xtra-Computing/JetFire — hybrid GPU–CPU attention with tiered precision
- Tiiny-AI/PowerInfer, SJTU-IPADS/PowerInfer-2 — hot/cold neuron tiering
- SGLang HiCache, TensorRT-LLM KVBM — tiered KV with admission and write policies
- Yunwei Zheng — *OS-Level Challenges in LLM Inference and Optimizations*

Foundational
- Denning — *The working set model for program behavior* (CACM 1968)
