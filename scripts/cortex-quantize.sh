#!/usr/bin/env bash
# cortex-quantize — reclaim the output matrix.
#
# Why this exists
# ---------------
# For models whose hidden dimension is not a multiple of 256 (Gemma-3-1B:
# hidden = 1152, and 1152 % 256 = 4.5), the K-quant formats are ILLEGAL for the
# token embedding and the output projection. llama.cpp silently falls back to
# Q8_0 for those tensors. On Gemma-3-1B that tensor is 262144 x 1152 = 302M
# params, i.e. ~40% of the file, pinned at 8 bits for a *format* reason rather
# than a sensitivity reason.
#
# Q4_0 uses QK = 32, and 1152 % 32 == 0, so it is legal. Re-quantizing just the
# output/embedding tensors to Q4_0 measured:
#
#   Gemma-3-1B Q4_K_M   762.49 MiB   tg128  84.30 tok/s
#   + Q4_0 out/embed    618.49 MiB   tg128  99.95 tok/s     (-19% bytes, +18.6%)
#
# This is NOT free: it is a 4-bit output matrix, so perplexity moves. Measure it
# before shipping (see docs/PERF_NOTES.md).
#
# Usage: cortex-quantize.sh <in.gguf> <out.gguf> [type] [out_type]
set -euo pipefail

IN="${1:?usage: cortex-quantize.sh <in.gguf> <out.gguf> [type] [out_type]}"
OUT="${2:?usage: cortex-quantize.sh <in.gguf> <out.gguf> [type] [out_type]}"
TYPE="${3:-Q4_K_M}"
OUT_TYPE="${4:-q4_0}"

BIN="$(dirname "$0")/../build/bin/llama-quantize"
[ -x "$BIN" ] || { echo "error: $BIN not built (cmake --build build --target llama-quantize)" >&2; exit 1; }

echo "in : $IN ($(du -h "$IN" | cut -f1))"
"$BIN" --allow-requantize \
       --output-tensor-type "$OUT_TYPE" \
       --token-embedding-type "$OUT_TYPE" \
       "$IN" "$OUT" "$TYPE"
echo "out: $OUT ($(du -h "$OUT" | cut -f1))"
