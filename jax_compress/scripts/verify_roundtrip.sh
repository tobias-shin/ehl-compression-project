#!/usr/bin/env bash
# Round-trip md5 check for compressed.dat against an original input.
#
# Usage:
#   ./verify_roundtrip.sh <original_input>
#   ./verify_roundtrip.sh data/enwik8
#
# Assumes ``data/compressed.dat`` is the compressed bytestream produced
# by run_papermill.py --mode compress, and that the matching --mode
# decompress run has already been done. Reads:
#   data/<original>         -- the file we encoded
#   data/decompressed.dat   -- AC-decoded preprocessed-form bytes (for
#                              --preprocess nncp, this is the
#                              NNCP-tokenized representation, NOT the
#                              original bytes)
#   data/final.dat          -- the unpreprocessed output, which IS what
#                              should match the original
#
# Gotcha that bit us once (commit 162e781 cleanup): for
# --preprocess nncp runs, ``decompressed.dat`` is an intermediate (~60-80%
# of original size, NNCP-encoded). The actual reconstructed original
# lives in ``final.dat`` after the trailing ``!./nncp/preprocess d``
# pipeline step in the Decompression cell. ALWAYS compare md5sum
# against ``data/final.dat`` for --preprocess nncp; against
# ``data/decompressed.dat`` only for --preprocess none.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <original_input>" >&2
  exit 1
fi

ORIG="$1"
if [ ! -f "$ORIG" ]; then
  echo "error: original input file not found: $ORIG" >&2
  exit 1
fi

if [ -f "data/final.dat" ]; then
  # NNCP-preprocessed run: compare against final.dat (unpreprocessed)
  echo "=== Comparing original to data/final.dat (nncp un-preprocessed output) ==="
  md5sum "$ORIG" data/final.dat
elif [ -f "data/decompressed.dat" ]; then
  # No-preprocess run: compare against decompressed.dat directly
  echo "=== Comparing original to data/decompressed.dat (no-preprocess decode) ==="
  md5sum "$ORIG" data/decompressed.dat
else
  echo "error: neither data/final.dat nor data/decompressed.dat exists." >&2
  echo "       Did you run --mode decompress?" >&2
  exit 1
fi
