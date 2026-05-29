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

# JAX reference points (from Byron Knoll's jax-compress README)
#   enwik8: 15,505,441 bytes -> 1.2404 bpc (no preprocessing dict).
#   enwik9: 113,393,442 bytes compressed + 80,040 byte dict = 113,473,482 bytes
#          -> 0.9078 bpc total. (An earlier version of this list had 0.9114
#          which was an arithmetic error on my part.)
JAX_REF = [
    (100_000_000, 1.2404, 'jax-compress enwik8'),
    (1_000_000_000, 0.9078, 'jax-compress enwik9'),
]

# NNCP reference points -- LTCB-verified (Mahoney's Large Text Compression
# Benchmark, http://mattmahoney.net/dc/text.html). NNCP v2.1 is the version
# vendored at jax_compress/nncp/preprocess.c, released 2021-02-06.
#   nncp v2.1 enwik8: 15,020,691 bytes -> 1.2017 bpc (compressed file only).
#   nncp v2.1 enwik9: 112,219,309 bytes (compressed) + 100,046 bytes dict
#                    = 112,319,355 bytes total. The 0.8978 bpc figure used
#                    here is compressed-only, matching the convention used
#                    for our own runs in HYBRID_DATA / XL_DATA / DATA.
# Earlier "NNCP-base ~1.0 bpc" approximations (commit e2cdbb4) were simply
# wrong -- LTCB has no version of NNCP at ~1.0 bpc on enwik8.
NNCP_REF = [
    (100_000_000, 1.2017, 'nncp v2.1 enwik8'),
    (1_000_000_000, 0.8978, 'nncp v2.1 enwik9'),
]

# Hybrid backend (model_type=hybrid -- LSTM + Transformer-XL geometric-mean
# ensemble; both submodels train independently per step, AC sees the combined
# distribution). Filled in as runs complete.
HYBRID_DATA = [
    (    10_000, 3.8000, 'enwik4', 'fp32', 'none'),
    (   100_000, 2.8269, 'enwik5', 'fp32', 'nncp'),
    ( 1_000_000, 2.0909, 'enwik6', 'fp32', 'nncp'),
    (10_000_000, 1.6465, 'enwik7', 'fp32', 'nncp'),
    # enwik8 hybrid: 14h 16min, --use-bf16 --mode compress, default hparams.
    # First size at which the hybrid beats the LSTM solo (1.27 vs 1.29).
    # Round-trip not verified (--mode compress); trust extrapolated from
    # enwik4-6 hybrid round-trip md5 matches.
    (100_000_000, 1.2723, 'enwik8', 'bf16', 'nncp'),
]

# Hybrid + learned mixer (model_type=hybrid + --use-learned-mixer). Tiny MLP
# (~320 params) produces per-step softmax weights from per-submodel
# confidence features (entropy + max log-prob). Equal-weight ensemble is in
# the mixer's hypothesis class so it can never be strictly worse than
# HYBRID_DATA at convergence; should win modestly at every scale and more
# meaningfully where one submodel is much better than the other.
HYBRID_MIXER_DATA = [
    (    10_000, 3.6656, 'enwik4', 'fp32', 'none'),
    (   100_000, 2.6597, 'enwik5', 'fp32', 'nncp'),
    ( 1_000_000, 1.9934, 'enwik6', 'fp32', 'nncp'),
    # enwik7: --mode compress only; round-trip md5 verified at enwik4-6.
    (10_000_000, 1.6018, 'enwik7', 'fp32', 'nncp'),
    # enwik8: 14h 46min, --use-bf16 --mode compress, default hparams.
    # Beats LSTM by 0.029 bpc (largest gap of the sweep so far). Round-trip
    # md5 not verified at this size (--mode compress); bf16 round-trip is
    # verified at enwik4-6 mixer runs.
    (100_000_000, 1.2626, 'enwik8', 'bf16', 'nncp'),
]

# Hybrid + mixer + Tier 1 LR-schedule pull-in (transitions 50K/150K instead
# of NNCP's never-firing 341K/3.13M). Committed as default in 7407e2f, then
# silently disabled by the --xl-lr-schedule CLI default bug, then re-enabled
# by b0cf21c, then reverted in 0e52492 after this enwik8 regression.
# clip_xl=0.25 + adam_eps_xl=1e-9 (the other Tier 1 changes) are kept since
# they're verified neutral here. New schedule was effectively a no-op at
# enwik4-6 (transitions don't fire), partially fired at enwik7 (50K
# transition only, last ~28K of ~78K steps), and fired both transitions at
# enwik8 -- where it cost +0.0089 bpc as the 5e-6 floor undertrains late
# tokens.
HYBRID_MIXER_T1_DATA = [
    (    10_000, 3.6656, 'enwik4', 'fp32', 'none'),
    (   100_000, 2.6597, 'enwik5', 'fp32', 'nncp'),
    ( 1_000_000, 1.9930, 'enwik6', 'fp32', 'nncp'),
    (10_000_000, 1.6014, 'enwik7', 'fp32', 'nncp'),
    # enwik8: 14h 27min, --use-bf16 --mode compress. +0.0089 bpc vs mixer_v2.
    (100_000_000, 1.2715, 'enwik8', 'bf16', 'nncp'),
]

# Hybrid + mixer + adaptive PPM-style n-gram (order=3, Laplace alpha=0.01) as
# a 3rd submodel through the mixer. Pure stats: no params, no gradients;
# diversity comes from being structurally orthogonal to the two neural
# submodels. Combined with Tier 1 clip_xl=0.25 + adam_eps_xl=1e-9 but with
# the OLD never-firing LR schedule (these runs predate the b0cf21c CLI fix).
# Helped at enwik7 (-0.020 vs mixer baseline; the strongest gain we've seen
# at any scale) but hurt at enwik8 (+0.018). Submodel + plumbing reverted in
# d4b85c7 since it didn't generalize to the largest scale.
HYBRID_T1_NG_DATA = [
    # enwik7: 1.5815 -- best result at this scale across all experiments.
    (10_000_000, 1.5815, 'enwik7', 'fp32', 'nncp'),
    # enwik8: 1.2805 -- regression vs mixer_v2 (1.2626). About half of the
    # +0.018 gap is from the ngram, half from the Tier 1 LR pull-in (this run
    # had the buggy CLI default that silently used OLD schedule, so the
    # actual confound here is just clip+eps + ngram, not LR pull-in).
    (100_000_000, 1.2805, 'enwik8', 'bf16', 'nncp'),
]

# Hybrid + mixer + Transformer-XL submodel scaled up to NNCP-large hparams
# (d_model=1024, d_head=128, d_inner=4096; 154M XL params vs 44M base).
# Other NNCP-large hparams adopted: batch_size=64 (was 128), retrain_batch_size=32
# (was 256), xl_lr_schedule "0:4.0e-5 341105:1.3e-5 3134681:4.0e-6" (lower start
# LR for the bigger model), matching xl_retrain_lr_schedule. Kept our
# n_words=8192 preprocess (NOT NNCP-large's 4096) -- "option (b)" of two
# possible XL-large flavors. LSTM submodel and mixer unchanged.
HYBRID_MIXER_XL_LARGE_DATA = [
    # enwik8: 86h 31min, --use-bf16 --mode compress. Total params 295M
    # (LSTM 142M + XL-large 154M + mixer 0.5K). 5.9x the wall of mixer_rb10m
    # (14h45m) but improves bpc by 0.0099. Halves the gap to jax-compress
    # (1.2404 ref) from +0.018 to +0.008. Round-trip not verified at enwik8
    # (--mode compress); enwik4 dry run with same XL-large config round-trip
    # md5-matches.
    (100_000_000, 1.2488, 'enwik8', 'bf16', 'nncp'),
]

# Hybrid + mixer + XL-large + the two remaining NNCP-large knobs that
# HYBRID_MIXER_XL_LARGE_DATA didn't ship: n_words=4096 (was 8192) and
# retrain_block_len=15M (was 10M). Tests whether the full NNCP-large config
# closes more of the gap to nncp v2.1, on top of the model-size win.
# Transformer-XL solo (no LSTM, no mixer) with NNCP-large hparams: same
# d_model=1024/d_head=128/d_inner=4096/n_layer=12, same NNCP-large LR
# schedule, same batch_size=64, same retrain_block_len=15M,
# n_words=4096. Pure single-model run for direct head-to-head with NNCP
# v2.1 (1.2017 enwik8) -- intended to isolate "what's our XL submodel
# alone worth, before the mixer ensembling effect". Result at enwik8 is
# 1.3388 -- the LSTM submodel was meaningfully contributing (-0.091 in
# the hybrid vs solo at enwik8) AND there's a substantial 0.137 bpc gap
# from our XL alone to NNCP's XL alone at the same architecture spec,
# suggesting a numerics/training-loop fidelity gap (fp16 vs bf16,
# deterministic mode, or training-loop drift).
TRANSFORMER_XL_LARGE_SOLO_DATA = [
    (    10_000, 4.0496, 'enwik4', 'bf16', 'none'),
    (   100_000, 3.2240, 'enwik5', 'bf16', 'nncp'),
    ( 1_000_000, 2.3516, 'enwik6', 'bf16', 'nncp'),
    (10_000_000, 1.7630, 'enwik7', 'bf16', 'nncp'),
    (100_000_000, 1.3388, 'enwik8', 'bf16', 'nncp'),
]

# Same XL-large solo config but with fp16 mixed precision (--use-fp16
# --no-bf16). NNCP uses fp16 (per nncp_enwik_*.sh: --fp16); this run
# tests whether bf16-vs-fp16 explains the 0.137 bpc gap from our XL
# solo to NNCP v2.1's. Conclusion: it does not -- fp16 lands within
# 0.001 bpc of bf16 at every scale.
TRANSFORMER_XL_LARGE_SOLO_FP16_DATA = [
    (    10_000, 4.0496, 'enwik4', 'fp16', 'none'),
    (   100_000, 3.2241, 'enwik5', 'fp16', 'nncp'),
    ( 1_000_000, 2.3521, 'enwik6', 'fp16', 'nncp'),
    (10_000_000, 1.7636, 'enwik7', 'fp16', 'nncp'),
    (100_000_000, 1.3387, 'enwik8', 'fp16', 'nncp'),
]

HYBRID_MIXER_XL_LARGE_FULL_DATA = [
    # enwik4: 37s, --preprocess none (file too small for an n-words=4096
    # dictionary). -0.1624 bpc vs mixer baseline (3.6656 -> 3.5032). The
    # 154M XL submodel is wildly over-parameterised for 78 streaming steps
    # but the extra capacity still helps per step at this regime.
    (    10_000, 3.5032, 'enwik4', 'bf16', 'none'),
    # enwik5: 3.7min. -0.0856 vs baseline (2.6597 -> 2.5741).
    (   100_000, 2.5741, 'enwik5', 'bf16', 'nncp'),
    # enwik6: 25.4min. -0.0500 vs baseline (1.9934 -> 1.9434).
    ( 1_000_000, 1.9434, 'enwik6', 'bf16', 'nncp'),
    # enwik7: 4h 42min. -0.0262 vs baseline (1.6018 -> 1.5756). Lowest bpc
    # on this file in the codebase, beats the prior best (t1_ng at 1.5815).
    (10_000_000, 1.5756, 'enwik7', 'bf16', 'nncp'),
    # enwik8: 114h 55min, --use-bf16 --mode compress. -0.0011 bpc vs
    # xl_large alone -- marginal but real. Wall is 1.33x xl_large from the
    # 20%-more-tokens overhead of n_words=4096. Round-trip not verified at
    # enwik8 (--mode compress).
    (100_000_000, 1.2477, 'enwik8', 'bf16', 'nncp'),
]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.normpath(os.path.join(THIS_DIR, '..', 'data', 'bpc_vs_size.png'))


def _dedup_pts(rows):
    """Collapse duplicate sizes: latest entry wins."""
    seen = {}
    for s, b, l, *_ in rows:
        seen[s] = (s, b, l)
    return [seen[s] for s in sorted(seen)]


def _plot_lstm(ax):
    fp32_pts = [(s, b, l) for (s, b, l, p, _) in DATA if p == 'fp32']
    if fp32_pts:
        xs, ys, _ = zip(*fp32_pts)
        ax.plot(xs, ys, 'o-', color='#1f77b4', label='torch LSTM (fp32)', markersize=8)
        for x, y, l in fp32_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6), fontsize=9)


def _plot_transformer_xl(ax):
    xl_pts = _dedup_pts(XL_DATA)
    if xl_pts:
        xs, ys, _ = zip(*xl_pts)
        ax.plot(xs, ys, 'D-', color='#d62728', label='torch transformer_xl (fp32)',
                markersize=8)
        for x, y, l in xl_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6),
                        fontsize=9, color='#d62728')


def _plot_hybrid(ax):
    hyb_pts = _dedup_pts(HYBRID_DATA)
    if hyb_pts:
        xs, ys, _ = zip(*hyb_pts)
        ax.plot(xs, ys, '*-', color='#8c564b', label='torch hybrid (LSTM + xl, equal-weight)',
                markersize=12)
        for x, y, l in hyb_pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6),
                        fontsize=9, color='#8c564b')


def _plot_hybrid_mixer(ax):
    pts = _dedup_pts(HYBRID_MIXER_DATA)
    if pts:
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, 's-', color='#e377c2',
                label='torch hybrid (LSTM + xl, learned mixer)',
                markersize=8)
        for x, y, l in pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, -14),
                        fontsize=9, color='#e377c2')


def _plot_hybrid_mixer_t1(ax):
    pts = _dedup_pts(HYBRID_MIXER_T1_DATA)
    if pts:
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, 'x--', color='#ff7f0e',
                label='torch hybrid + mixer + Tier 1 LR pull-in (reverted)',
                markersize=8, alpha=0.8)


def _plot_hybrid_t1_ng(ax):
    pts = _dedup_pts(HYBRID_T1_NG_DATA)
    if pts:
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, 'P--', color='#bcbd22',
                label='torch hybrid + mixer + n-gram submodel (reverted)',
                markersize=9, alpha=0.8)


def _plot_hybrid_mixer_xl_large(ax):
    pts = _dedup_pts(HYBRID_MIXER_XL_LARGE_DATA)
    if pts:
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, 'h-', color='#17becf',
                label='torch hybrid + mixer + XL-large (NNCP-large XL hparams)',
                markersize=10, markerfacecolor='#17becf')
        for x, y, l in pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, -14),
                        fontsize=9, color='#17becf')


def _plot_hybrid_mixer_xl_large_full(ax):
    pts = _dedup_pts(HYBRID_MIXER_XL_LARGE_FULL_DATA)
    if pts:
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, 'p-', color='#9edae5',
                label='torch hybrid + mixer + XL-large + n_words=4096 (full NNCP-large)',
                markersize=10, markerfacecolor='#9edae5')
        for x, y, l in pts:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, 6),
                        fontsize=9, color='#17becf')


def _plot_transformer_xl_large_solo(ax):
    pts = _dedup_pts(TRANSFORMER_XL_LARGE_SOLO_DATA)
    if pts:
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, 'D--', color='#ff9896',
                label='torch transformer_xl-large solo (bf16, NNCP-large hparams)',
                markersize=8, alpha=0.85)


def _plot_transformer_xl_large_solo_fp16(ax):
    pts = _dedup_pts(TRANSFORMER_XL_LARGE_SOLO_FP16_DATA)
    if pts:
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, 'd:', color='#c49c94',
                label='torch transformer_xl-large solo (fp16, NNCP-large hparams)',
                markersize=8, alpha=0.85)


def _plot_refs(ax):
    if JAX_REF:
        xs, ys, _ = zip(*JAX_REF)
        ax.plot(xs, ys, '^--', color='#2ca02c', label='jax-compress (Knoll, reference)',
                markersize=8, alpha=0.7)
        for x, y, l in JAX_REF:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(-8, -14),
                        fontsize=9, color='#2ca02c')
    if NNCP_REF:
        xs, ys, _ = zip(*NNCP_REF)
        ax.plot(xs, ys, 'v--', color='#9467bd', label='nncp v2.1 (LTCB, reference)',
                markersize=8, alpha=0.7)
        for x, y, l in NNCP_REF:
            ax.annotate(l, (x, y), textcoords='offset points', xytext=(8, -14),
                        fontsize=9, color='#9467bd')


def _format_axis(ax, title):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('file size (bytes)')
    ax.set_ylabel('bpc (bits per byte of original) [log scale]')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    # Entropy floor annotation
    ax.axhline(y=0.91, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('≈ entropy floor (~0.9 bpc)', xy=(2e4, 0.95), color='gray', fontsize=8)
    # On a log y-axis, matplotlib defaults to scientific notation; force
    # plain decimal labels so bpc values like 1.27 read as 1.27 not 10^0.1.
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())


def _enwik8_detail(ax):
    """Single-scale detail panel: every variant's enwik8 bpc as a horizontal
    band, sorted best-to-worst, with the ablations rendered as outlined dots
    so they read as 'tried-and-rejected' relative to the solid headline
    points. Linear y-axis at full resolution -- ~0.02 bpc differences that
    are invisible on the log-log overview become legible here."""
    pts = []
    def add(label, bpc, color, marker, fill=True, weight='normal'):
        pts.append((label, bpc, color, marker, fill, weight))
    add('nncp v2.1', 1.2017, '#9467bd', 'v', True, 'bold')
    add('jax-compress (Knoll)', 1.2404, '#2ca02c', '^', True, 'bold')
    add('hybrid + mixer + full NNCP-large', 1.2477, '#9edae5', 'p', True, 'bold')
    add('hybrid + mixer + XL-large', 1.2488, '#17becf', 'h', True, 'bold')
    add('hybrid + mixer + retrain-block 10M', 1.2587, '#e377c2', 's', True)
    add('hybrid + mixer (mixer_v2)', 1.2626, '#e377c2', 's', True)
    add('hybrid (equal-weight)', 1.2723, '#8c564b', '*', True)
    add('mixer + Tier 1 LR pull-in', 1.2715, '#ff7f0e', 'x', False)
    add('mixer + ngram (t1_ng)', 1.2805, '#bcbd22', 'P', False)
    add('LSTM solo', 1.2918, '#1f77b4', 'o', True)
    add('XL-large solo (bf16)', 1.3388, '#ff9896', 'D', True)
    add('XL-large solo (fp16)', 1.3387, '#c49c94', 'd', True)
    add('Transformer-XL-base solo', 1.3734, '#d62728', 'D', True)
    pts.sort(key=lambda r: r[1])
    for i, (label, bpc, color, marker, fill, weight) in enumerate(pts):
        y = len(pts) - 1 - i
        ms_kwargs = dict(markersize=10) if fill else dict(markersize=11,
                                                          markerfacecolor='white',
                                                          markeredgewidth=2)
        ax.plot([bpc], [y], marker=marker, color=color, linestyle='', **ms_kwargs)
        ax.annotate(f'{label}', (bpc, y), textcoords='offset points',
                    xytext=(12, 0), fontsize=9, va='center', color=color,
                    fontweight=weight)
        ax.annotate(f'{bpc:.4f}', (bpc, y), textcoords='offset points',
                    xytext=(-8, 0), fontsize=8, va='center', ha='right',
                    color='dimgray')
    # Reference span: nncp v2.1 (1.2017) -> our current best (1.2477 xl_large_full)
    ax.axvspan(1.2017, 1.2477, alpha=0.08, color='gray')
    ax.set_xlabel('bpc (lower is better)')
    ax.set_yticks([])
    ax.set_xlim(1.18, 1.42)
    ax.set_ylim(-0.7, len(pts) - 0.3)
    ax.set_title('enwik8 detail (linear bpc)')
    ax.grid(True, axis='x', alpha=0.3)
    ax.annotate('open marker = reverted experiment', xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=7, color='gray',
                style='italic')


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={'width_ratios': [1.4, 1]})
    ax_main, ax_detail = axes
    _plot_lstm(ax_main)
    _plot_transformer_xl(ax_main)
    _plot_hybrid(ax_main)
    _plot_hybrid_mixer(ax_main)
    _plot_hybrid_mixer_t1(ax_main)
    # _plot_hybrid_t1_ng(ax_main)  -- only enwik7/8 measured; omitted from
    #     the curve panel because it doesn't have a full enwik4-8 sweep.
    #     Still listed in the enwik8 detail panel.
    # _plot_hybrid_mixer_xl_large(ax_main)  -- only enwik8 measured (the
    #     n_words=8192 "option b" intermediate before the full NNCP-large
    #     sweep). Single-point curve; same reason as above.
    _plot_hybrid_mixer_xl_large_full(ax_main)
    _plot_transformer_xl_large_solo(ax_main)
    _plot_transformer_xl_large_solo_fp16(ax_main)
    _plot_refs(ax_main)
    _format_axis(ax_main, 'Compression rate vs file size — torch_compress')
    _enwik8_detail(ax_detail)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
