#!/usr/bin/env python3
"""Plot bpc vs file size for the torch_compress sweep.

Edit the DATA list to add/update points as new sweep runs complete. The
XL_DATA list is the parallel series for the transformer_xl backend (Phase 4
of the NNCP swap-in); fill it in as the corresponding runs complete.

Outputs a PNG to data/bpc_vs_size.png.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# torch_compress LSTM sweep -- (size_bytes, bpc, label, precision, preprocess)
# NOTE: every LSTM run is fp32 in practice -- the use_bf16 notebook parameter
# was advertised but never wired into any code path on the LSTM side, so the
# previously-labelled "bf16" rows were actually running fp32. The two enwik7
# rows (1.6174, 1.6159) reflect two separate fp32 runs with slightly
# different bpc due to seed / run-time variance.
DATA = [
    (    10_000, 3.6688, 'enwik4', 'fp32', 'none'),
    (   100_000, 2.6665, 'enwik5', 'fp32', 'nncp'),
    ( 1_000_000, 1.9992, 'enwik6', 'fp32', 'nncp'),
    (10_000_000, 1.6174, 'enwik7', 'fp32', 'nncp'),  # was mislabeled "bf16"
    (10_000_000, 1.6159, 'enwik7', 'fp32', 'nncp'),
    (100_000_000, 1.2918, 'enwik8', 'fp32', 'nncp'),
]

# torch_compress + Transformer-XL backend (model_type=transformer_xl,
# NNCP-base hparams). Fill in as runs complete -- enwik4 point landed during
# the Phase 4A sanity check (--preprocess none, since NNCP preprocessing
# isn't useful at 10 KB and at-scale comparison uses byte-level there too).
XL_DATA = [
    # enwik4-7 are fp32. bf16 mixed precision is round-trip safe (commit
    # 7611a88) but at these file sizes is ~8% slower than fp32 on GH200 with
    # no bpc benefit (differences below 0.001). Run with --use-bf16 to opt in.
    # md5 round-trip verified in each papermill Validation cell.
    (    10_000, 4.4360, 'enwik4', 'fp32', 'none'),
    (   100_000, 3.4981, 'enwik5', 'fp32', 'nncp'),
    ( 1_000_000, 2.4986, 'enwik6', 'fp32', 'nncp'),
    (10_000_000, 1.8704, 'enwik7', 'fp32', 'nncp'),
    # enwik8 is bf16 (--use-bf16 --mode compress) with NNCP-aligned data-
    # pipeline hparams: batch_size=64, n_words=16384, retrain_block_len=10M,
    # retrain_batch_size=32, retrain_period_schedule="0:7813". Round-trip not
    # verified (--mode compress skipped decode); bf16 round-trip is verified
    # at enwik4-6 so extrapolation is plausible but not formally checked.
    # The original-defaults run at this size hit 1.3900 bpc in 3h52m;
    # NNCP-alignment improved to 1.3734 in 6h40m -- only 0.017 bpc lower
    # despite 2x wall clock. The remaining gap to NNCP-base's ~1.0 bpc is
    # likely a mix of: clip=4.0 (vs NNCP 0.25), bf16 vs fp16 numerics, the
    # absence of NNCP-style block_len mems-reset, and the LR schedule
    # transition firing very late in our 344K-step run.
    (100_000_000, 1.3734, 'enwik8', 'bf16', 'nncp'),
]

# JAX reference points (from the jax-compress README)
JAX_REF = [
    (100_000_000, 1.2404, 'jax enwik8'),
    (1_000_000_000, 0.9114, 'jax enwik9'),  # 113,393,442 / 1e9 * 8
]

# NNCP v2 reference (Bellard 2021, base config) -- the bar we want to beat.
# enwik8: ~9 hours on RTX 3090, fp16. Approximate published bpc.
NNCP_REF = [
    (100_000_000, 1.0066, 'nncp-base enwik8'),  # Bellard reports ~1.0 bpc base
    (1_000_000_000, 0.9943, 'nncp-base enwik9'),  # ~0.99 bpc base config
]

# Hybrid backend (model_type=hybrid -- LSTM + Transformer-XL geometric-mean
# ensemble; both submodels train independently per step, AC sees the combined
# distribution). Filled in as runs complete.
HYBRID_DATA = [
    (    10_000, 3.8000, 'enwik4', 'fp32', 'none'),
    (   100_000, 2.8269, 'enwik5', 'fp32', 'nncp'),
    ( 1_000_000, 2.0909, 'enwik6', 'fp32', 'nncp'),
    (10_000_000, 1.6465, 'enwik7', 'fp32', 'nncp'),
    # (100_000_000, ?, 'enwik8', 'bf16', 'nncp'),
]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.normpath(os.path.join(THIS_DIR, '..', 'data', 'bpc_vs_size.png'))


def _dedup_pts(rows):
    """Collapse duplicate sizes: latest entry wins."""
    seen = {}
    for s, b, l, p, _ in rows:
        seen[s] = (s, b, l)
    return [seen[s] for s in sorted(seen)]


def _plot_lstm(ax):
    fp32_pts = [(s, b, l) for (s, b, l, p, _) in DATA if p == 'fp32']
    if fp32_pts:
        xs, ys, ls = zip(*fp32_pts)
        ax.plot(xs, ys, 'o-', color='#1f77b4', label='torch LSTM (fp32)', markersize=8)
        for x, y, l in fp32_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6), fontsize=9)


def _plot_transformer_xl(ax):
    xl_pts = _dedup_pts(XL_DATA)
    if xl_pts:
        xs, ys, ls = zip(*xl_pts)
        ax.plot(xs, ys, 'D-', color='#d62728', label='torch transformer_xl (fp32)',
                markersize=8)
        for x, y, l in xl_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6),
                        fontsize=9, color='#d62728')


def _plot_hybrid(ax):
    hyb_pts = _dedup_pts(HYBRID_DATA)
    if hyb_pts:
        xs, ys, ls = zip(*hyb_pts)
        ax.plot(xs, ys, '*-', color='#8c564b', label='torch hybrid (LSTM + xl)',
                markersize=12)
        for x, y, l in hyb_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6),
                        fontsize=9, color='#8c564b')


def _plot_refs(ax):
    if JAX_REF:
        xs, ys, ls = zip(*JAX_REF)
        ax.plot(xs, ys, '^--', color='#2ca02c', label='jax (bf16, reference)',
                markersize=8, alpha=0.7)
        for x, y, l in JAX_REF:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(-8, -14),
                        fontsize=9, color='#2ca02c')
    if NNCP_REF:
        xs, ys, ls = zip(*NNCP_REF)
        ax.plot(xs, ys, 'v--', color='#9467bd', label='nncp-base (fp16, reference)',
                markersize=8, alpha=0.7)
        for x, y, l in NNCP_REF:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, -14),
                        fontsize=9, color='#9467bd')


def _format_axis(ax, title):
    ax.set_xscale('log')
    ax.set_xlabel('file size (bytes)')
    ax.set_ylabel('bpc (bits per byte of original)')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')
    # Entropy floor annotation
    ax.axhline(y=0.91, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('≈ entropy floor (~0.9 bpc)', xy=(2e4, 0.95), color='gray', fontsize=8)


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot_lstm(ax)
    _plot_transformer_xl(ax)
    _plot_hybrid(ax)
    _plot_refs(ax)
    _format_axis(ax, 'Compression rate vs file size — torch_compress (LSTM, Transformer-XL, Hybrid)')
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
