# Performance notes: what moved, what didn't, and why

All numbers measured on **Apple M1 Pro** (8-core CPU / 14-core GPU, 16 GB unified,
200 GB/s spec peak), macOS 26.6.2, `-DGGML_METAL=ON`, unless stated otherwise.
Decode is `tg128` at batch 1 unless stated.

This file exists so the same dead ends are not re-explored. Every "no" below is a
measurement, not an opinion.

---

## 1. The one change that worked: reclaim the output matrix

Gemma-3-1B has `hidden = 1152`. The K-quant formats require `ne00 % 256 == 0`,
and `1152 % 256 = 4.5`, so **Q4_K/Q2_K are illegal for the token embedding and
output projection**. llama.cpp silently falls back to Q8_0 for those tensors.
That tensor is `262144 x 1152 = 302M` params — about 40% of the file — pinned at
8 bits for a *format* reason, not a sensitivity reason.

Q4_0 uses `QK = 32`, and `1152 % 32 == 0`, so it is legal.

| model | size | pp512 | tg128 |
|---|---|---|---|
| Gemma-3-1B Q4_K_M | 762.49 MiB | 1704.49 | 84.30 |
| + output/embedding at Q4_0 | **618.49 MiB** | 1708.97 | **99.95** |

**−19% bytes, +18.6% decode.** Throughput tracks bytes almost exactly, which is
the signature of a purely bandwidth-bound workload.

**Quality cost, measured** (wikitext-2, `-c 512`):

| model | PPL | delta |
|---|---|---|
| Q4_K_M (output at Q8_0) | 29.8848 ± 0.28913 | — |
| output/embedding at Q4_0 | 30.7590 ± 0.30025 | **+2.93%** |

So the trade is **+18.6% decode for +2.9% perplexity**. That is a good trade for
throughput-oriented use and a bad one for quality-critical use — it is a choice,
not a free win. `scripts/cortex-quantize.sh` applies it.

---

## 2. The roofline, and why 2.7x is not available

At 618 MB/token and 100 tok/s the engine moves **61.8 GB/s**, i.e. ~31% of the
200 GB/s spec peak. That looks like a 3x opportunity. It is not:

- M1 Pro's *achievable* STREAM-class bandwidth is ~150 GB/s (~75% of spec).
- 4-bit batch-1 decode on Apple GPUs normally lands at **46–59% of that GEMV
  wall**, because dequantisation is ALU work interleaved with the loads.
- 31% of spec is therefore ~41% of achievable — the normal operating point, not
  an anomaly.
- The best hand-written native Metal runtime (BaseRT) beats llama.cpp by
  **1.15–1.56x**, and its own paper attributes the win to *CPU-side* overhead
  (zero-allocation decode loop, low-overhead dispatch, amortised CPU–GPU sync),
  not to faster matvec kernels.

**Realistic ceiling for this shape: ~1.3–1.6x, and it requires a kernel/dispatch
rewrite, not a flag.**

---

## 3. Dead ends (measured, do not retry)

| change | result | verdict |
|---|---|---|
| Disable K-quant small-batch (`mul_mv_ext`) path | B=8: 662 → 589 tok/s | **worse**; the published M4/8B result does not transfer |
| `N_R0_Q4_K` / `N_SG_Q4_K` sweep (2,2 / 4,2 / 4,4 / 2,4 / 8,2 / 8,4) | 164.8 / 159.5 / 162.6 / 164.2 / 155.8 / 156.3 | baseline (2,2) already optimal |
| Backend sampling (`-bs`) | 92–94 vs 82–98 | noise |
| n-gram speculation (`--spec-type ngram-simple`) | 99.2 vs 99.6 | noise on non-copy prompts |
| `GGML_METAL_NO_RESIDENCY=1` | +0.5% | noise |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=60` | +1.3% | marginal |
| `GGML_METAL_SHARED_BUFFERS_DISABLE=1` | −20% | **much worse** |
| `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1` | −1.9% | worse |
| `-t 4 / 6 / 8` | 110.5 / 108.8 / 111.0 | noise |
| `-fa on / off` | 109.9 / 105.9 | default already on |
| `-nkvo 1` | 28.7 | **catastrophic** |
| `-b 512 -ub 128` / `-b 1024 -ub 256` | 99.9 / 100.1 | noise |
| Decode at depth 0 / 4096 / 16384 | 46.8 / 47.1 / 45.0 | flat — KV is not the bottleneck (sliding window) |

### Why the activation-re-read hypothesis was wrong

Each threadgroup re-reads the whole activation vector `y`. With `nr0=2, nsg=2`
that is 4 weight rows per threadgroup, so for `ne01=5120` there are 1280
threadgroups and `y` is re-read 1280 times — 25.6 MB of activation traffic
against 14.7 MB of weights. It looked like the dominant term.

It is not: `y` is 20 KB and lives in L2, so the re-reads never reach DRAM.
Raising `nr0*nsg` to cut them only costs occupancy, and every configuration
measured slower. **The activation traffic is an L2-bandwidth effect, not a DRAM
one, and it is not the bottleneck.**

---

## 4. What the output matrix costs, and the remaining lever

For a 1B model with a 262k vocabulary the output projection is ~30% of the model
and ~30% of batch-1 decode. It is the single largest non-weight term.

Three training-free ways to cut it, in order of evidence:

1. **Legal 4-bit format** (this change): −19% bytes, +18.6% decode, small PPL cost.
2. **Low-rank factorisation** `V x d -> V x r + r x d`: cuts bytes by `r/d`.
   Training-free, but needs engine support for the factored form and a measured
   rank/quality curve.
3. **Vocabulary pruning**: drop tokens that never occur for the workload. Large
   win on narrow tasks, unsafe for general chat.

Combined, the literature reports 10–13x fewer bytes/token for the output path,
all training-free. That is the only remaining order-of-magnitude lever, and it is
a model-side change, not a kernel one.

---

## 5. The real generic win: gated n-gram speculation

Decode is bandwidth-bound, so the only way to beat one-token-per-weight-read is
to get more than one token per weight read. Speculative decoding does exactly
that, and n-gram drafting needs **no draft model** — it proposes continuations by
matching the current suffix against the context, then the target verifies them in
one batched pass. Verification is exact: the output distribution is unchanged.

The catch is that a rejected draft still costs a verification pass. Measured on
Gemma-3-1B Q4_0, 120 tokens:

| task | base | ungated | gated |
|---|---|---|---|
| copy / rewrite | 97.7 | 267.7 (2.68x) | **261.8 (2.68x)** |
| code | 97.7 | 95.7 (-4%) | 99.9 (+2.2%) |
| chat | 99.0 | 98.6 (-1%) | 98.0 (-1%) |

So the win is large and real on copy-heavy work, and the loss elsewhere is the
cost of verifying drafts that get rejected.

**The gate.** Track a rolling acceptance rate over the last 16 rounds (halved when
full, so it follows the recent regime) and stop drafting below the break-even
point. The break-even is not a guess: at the measured 5x batch amortisation,
verifying `k=8` drafts costs about `1 + (k-1)/5 = 2.4x` a single token, so the
expected accepted-token yield must exceed 2.4. For a geometric acceptance model
that is `alpha ~= 0.6`. At `alpha=0.5` the yield is 1.99 against a 2.4 cost — a
0.83x slowdown.

This makes the feature safe to leave on: it finds the copy-heavy stretches of a
workload and gets out of the way elsewhere. Implemented for `ngram-simple` and
`ngram-map-k`; `ngram-mod` already had a different adaptive behaviour (it resets
its store on a low-acceptance streak but keeps drafting).

**Why this matters more than kernel tuning:** it is the only lever that changes
the bytes-per-token ratio rather than the efficiency of moving them.

---

## 6. Tokenization: two real wins, one dead end

Tokenization is not where inference time goes — for a 10k-token prompt it is
~0.3% of prefill — but it is pure CPU, so it is measurable in isolation and it
was leaving 1.5x on the table.

Profiling a 13 MB tokenize run (`sample`, 6 s) put **4034 of ~4880 active
samples in `tokenizer_st_partition`**. That function walked the whole text once
per special token, and a chat model has ~100 of them: O(special_tokens ×
text_length). Indexing the tokens by first byte and doing a single longest-match
pass is equivalent (the cache is sorted by descending length, so the first match
at a position is the longest) and removes the repeated scans.

The second win was lifetime, not algorithm: `llm_tokenizer_spm_session` was
constructed **inside the per-fragment loop**, so every fragment rebuilt a
vector, a priority queue and a map. Hoisting it and resetting between fragments
keeps the allocations. Swapping `rev_merge` from `std::map` to a hash map
removed a string comparison per tree level plus a rebalance per insert.

| stage | 1 MB, min of 10 |
|---|---|
| before | 0.91 s |
| one-pass partition | 0.68 s |
| + session reuse, hash map | 0.59 s |

**1.5x, byte-identical**: 295,216 tokens over the full wikitext-2 test set, same
md5 before and after.

### Dead end: a direct-mapped lookup cache

The remaining cost is the string-keyed vocab hash lookup (~47% of active
samples). A direct-mapped cache in front of it — the CPU-L1 idea — was
implemented and measured **slower at every size**:

| entries | 1 MB |
|---|---|
| 256 | 0.64 s |
| 1024 | 0.60 s |
| 2048 | 0.62 s |
| 4096 | 0.62 s |
| no cache | 0.59 s |

Most SPM probes are misses, so a miss paid the cache hash on top of the map
hash. Reverted; the source carries a comment saying not to re-add it without a
hit-rate measurement.

### Measurement caveat

The 1.60x figure that appeared in an earlier commit was measured while the
desktop app was idle. With it running (load average ~4 on 8 cores) the same
binary measures 0.59–0.61 s. The README now says 1.5x. **Take min-of-N, and
check `uptime` before believing a small delta.**

## 7. Reproducing

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build -j8 --target llama-bench llama-cli llama-quantize llama-perplexity

# the win
./scripts/cortex-quantize.sh model-Q4_K_M.gguf model-q4o.gguf Q4_K_M q4_0
./build/bin/llama-bench -m model-Q4_K_M.gguf -m model-q4o.gguf -p 512 -n 128 -r 3
```
