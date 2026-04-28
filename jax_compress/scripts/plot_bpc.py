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
    # NOTE: the bf16 transformer runs we did had a non-deterministic retrain
    # path (autocast + use_deterministic_algorithms(True) silently picked a
    # non-deterministic bf16 kernel during retrain) -- bytes did not round-
    # trip on enwik6+ even though the streaming-only enwik4/5 bf16 runs did.
    # Rolling back to the valid fp32 numbers until bf16 is fixed properly.
    (    10_000, 4.4360, 'enwik4', 'fp32', 'none'),
    (   100_000, 3.4981, 'enwik5', 'fp32', 'nncp'),
    ( 1_000_000, 2.4986, 'enwik6', 'fp32', 'nncp'),
    # (10_000_000, ?, 'enwik7', 'fp32', 'nncp'),
    # (100_000_000, ?, 'enwik8', 'fp32', 'nncp'),
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.normpath(os.path.join(THIS_DIR, '..', 'data', 'bpc_vs_size.png'))


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Torch sweep points, color-coded by precision
    fp32_pts = [(s, b, l) for (s, b, l, p, _) in DATA if p == 'fp32']
    bf16_pts = [(s, b, l) for (s, b, l, p, _) in DATA if p == 'bf16']

    if fp32_pts:
        xs, ys, ls = zip(*fp32_pts)
        ax.plot(xs, ys, 'o-', color='#1f77b4', label='torch LSTM (fp32)', markersize=8)
        for x, y, l in fp32_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6), fontsize=9)

    if bf16_pts:
        # Currently empty -- LSTM was never actually run in bf16 (use_bf16 was
        # a no-op on that path). Kept for forward compatibility if/when the
        # LSTM path gets real mixed-precision support.
        xs, ys, ls = zip(*bf16_pts)
        ax.plot(xs, ys, 's-', color='#ff7f0e', label='torch LSTM (bf16)', markersize=8)
        for x, y, l in bf16_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6), fontsize=9)

    # Transformer-XL backend (collapsing duplicate sizes; latest entry wins
    # so the user can append a refined fp16/bf16 run without removing the
    # earlier fp32 row).
    xl_pts = []
    seen = {}
    for s, b, l, p, _ in XL_DATA:
        seen[s] = (s, b, l)
    for s in sorted(seen):
        xl_pts.append(seen[s])
    if xl_pts:
        xs, ys, ls = zip(*xl_pts)
        ax.plot(xs, ys, 'D-', color='#d62728', label='torch transformer_xl',
                markersize=8)
        for x, y, l in xl_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6),
                        fontsize=9, color='#d62728')

    # JAX reference (single line connecting available reference points)
    if JAX_REF:
        xs, ys, ls = zip(*JAX_REF)
        ax.plot(xs, ys, '^--', color='#2ca02c', label='jax (bf16, reference)', markersize=8, alpha=0.7)
        for x, y, l in JAX_REF:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(-8, -14), fontsize=9, color='#2ca02c')

    # NNCP v2 reference (the published bar we're trying to beat)
    if NNCP_REF:
        xs, ys, ls = zip(*NNCP_REF)
        ax.plot(xs, ys, 'v--', color='#9467bd', label='nncp-base (fp16, reference)',
                markersize=8, alpha=0.7)
        for x, y, l in NNCP_REF:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, -14),
                        fontsize=9, color='#9467bd')

    ax.set_xscale('log')
    ax.set_xlabel('file size (bytes)')
    ax.set_ylabel('bpc (bits per byte of original)')
    ax.set_title('Compression rate vs file size — torch_compress (LSTM vs Transformer-XL)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')

    # Annotate the entropy floor for English text
    ax.axhline(y=0.91, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('≈ entropy floor (~0.9 bpc)', xy=(2e4, 0.95), color='gray', fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
