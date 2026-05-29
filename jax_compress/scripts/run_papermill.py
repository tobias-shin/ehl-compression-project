"""Drive torch_compress.ipynb via papermill for local benchmark runs.

Strips the Colab-only Setup Files cell (which would re-download enwik8 and
nncp.zip), the Mount Google Drive cell, and the Download Result cells. Injects
parameters via papermill's `parameters`-tagged cell.

Usage:
    python scripts/run_papermill.py \\
        --input data/enwik5 \\
        --no-bf16 \\
        --tb-run-name enwik5_fp32_nncp \\
        --output-nb runs/papermill/enwik5_fp32_nncp.ipynb
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time

import papermill as pm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(REPO, "torch_compress.ipynb")

# Cells whose `#@title` we should drop entirely before papermill execution.
# These either depend on Colab (drive/files) or would re-download datasets that
# already exist locally.
DROP_TITLES = {
    "#@title Mount Google Drive",
    "#@title Setup Files",
    "#@title Download Result",
    "#@title Close Runtime",
}


def prepare_notebook(src_path: str, dst_path: str) -> None:
    nb = json.load(open(src_path))
    kept = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            first = "".join(cell["source"]).splitlines()[0] if cell["source"] else ""
            if first.strip() in DROP_TITLES:
                continue
        kept.append(cell)
    nb["cells"] = kept
    # papermill needs a kernelspec + language_info; default to python3.
    md = nb.setdefault("metadata", {})
    md.setdefault("kernelspec", {"display_name": "Python 3", "name": "python3", "language": "python"})
    md.setdefault("language_info", {"name": "python"})
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(nb, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to file to compress, e.g. data/enwik5")
    ap.add_argument("--use-bf16", dest="use_bf16", action="store_true")
    ap.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    ap.set_defaults(use_bf16=False)
    ap.add_argument("--use-fp16", dest="use_fp16", action="store_true",
                    help="float16 mixed precision (NNCP-style). Takes "
                         "precedence over --use-bf16. Untested for "
                         "round-trip stability; verify with --mode both.")
    ap.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    ap.set_defaults(use_fp16=False)
    ap.add_argument("--use-deterministic", dest="use_deterministic",
                    action="store_true",
                    help="LTCB-style deterministic kernels (default).")
    ap.add_argument("--no-deterministic", dest="use_deterministic",
                    action="store_false",
                    help="Disable deterministic cuBLAS/cuDNN -- faster but "
                         "the AC may not be bit-exact across machines. NNCP "
                         "runs this way; use it to compare against NNCP.")
    ap.set_defaults(use_deterministic=True)
    ap.add_argument("--preprocess", default="nncp", choices=["nncp", "none"])
    ap.add_argument("--tb-run-name", required=True)
    ap.add_argument("--tb-logdir", default="data/tensorboard")
    ap.add_argument("--mode", default="both", choices=["compress", "decompress", "both"])
    ap.add_argument("--output-nb", required=True, help="path to write executed notebook")
    ap.add_argument("--save-compressed", default=None,
                    help="if set, copy data/compressed.dat to this path after the run")
    # ---- Transformer-XL --------------------------------------------------
    # Default model is the LSTM. Pass --model-type transformer_xl to route to
    # the NNCP-style streaming + retraining loop. The --xl-* flags default to
    # nncp_enwik_base.sh values; they are read only when model_type == "transformer_xl".
    ap.add_argument("--model-type", choices=["lstm", "transformer_xl", "hybrid"], default="lstm")
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=1024,
                    help="NNCP-large default (was 512 for nncp_enwik_base). "
                         "Beats base by 0.0099 bpc on enwik8 hybrid+mixer.")
    ap.add_argument("--d-head", type=int, default=128,
                    help="NNCP-large default (was 64 for nncp_enwik_base).")
    ap.add_argument("--d-inner", type=int, default=4096,
                    help="NNCP-large default (was 2048 for nncp_enwik_base).")
    ap.add_argument("--mem-len", type=int, default=160)
    ap.add_argument("--ext-tgt-len", type=int, default=31)
    ap.add_argument("--attn-type", type=int, default=1)
    ap.add_argument("--tied-r-bias", type=int, default=1)
    ap.add_argument("--use-gelu", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--dropatt", type=float, default=0.0)
    ap.add_argument("--init-std", type=float, default=0.013)
    ap.add_argument("--retrain-tgt-len", type=int, default=64)
    ap.add_argument("--retrain-mem-len", type=int, default=128)
    ap.add_argument("--xl-lr-schedule",
                    default="0:4.0e-5 341105:1.3e-5 3134681:4.0e-6",
                    help="NNCP-large default (was nncp_enwik_base's 7.9e-5 "
                         "schedule). Lower starting LR compensates for the "
                         "bigger XL model's higher gradient magnitude. The "
                         "341K/3.13M transitions don't fire at our enwik8 "
                         "budget; pulling them in to 50K/150K regressed "
                         "enwik8 by +0.0089 bpc and was reverted.")
    ap.add_argument("--xl-retrain-lr-schedule",
                    default="0:1.6e-4 13000:1.6e-4 93000:7.9e-5 163000:4.0e-5 1911300:1.3e-5",
                    help="NNCP-large default (was nncp_enwik_base values).")
    # ---- Data-pipeline hparams (override notebook defaults) -------------
    # These are the LSTM-tuned defaults baked into the notebook params cell.
    # NNCP's nncp_enwik_base.sh uses very different values for enwik8/9 --
    # exposing them here so transformer runs can match NNCP without editing
    # the notebook.
    ap.add_argument("--batch-size", type=int, default=None,
                    help="parallel streams. Notebook default is now 64 "
                         "(NNCP-large value; required for VRAM headroom on "
                         "the XL-large submodel). Override to 128 for "
                         "transformer_xl-base or LSTM-only.")
    ap.add_argument("--n-words", type=int, default=None,
                    help="NNCP preprocess vocab. Notebook default 8192. "
                         "NNCP-base for enwik8 uses 16384.")
    ap.add_argument("--min-freq", type=int, default=None,
                    help="NNCP preprocess minimum word frequency to enter "
                         "the dictionary. Notebook default 64. NNCP-large "
                         "uses 512 for enwik8 AND enwik9.")
    ap.add_argument("--retrain-block-len", type=int, default=None,
                    help="trailing chars retrain may sample from. Notebook "
                         "default is now 10000000 (NNCP-base value); the "
                         "original 100K was LSTM-tuned and underused the "
                         "context. 10M improved enwik8 hybrid+mixer by "
                         "-0.0039 bpc (1.2626 -> 1.2587). Pass an explicit "
                         "value here to override the notebook default.")
    ap.add_argument("--retrain-batch-size", type=int, default=None,
                    help="batch dim during retrain. Notebook default is now "
                         "32 (NNCP-large value); the original 256 was "
                         "LSTM-tuned and over-allocates VRAM for the "
                         "XL-large retrain pass.")
    ap.add_argument("--retrain-period-schedule", default=None,
                    help='step-units schedule for retrain period. Notebook '
                         'default "0:1001 200000:5001". NNCP-base equivalent '
                         'with batch_size=64: "0:7813".')
    # ---- Hybrid learned mixer (only used when model_type == "hybrid") ----
    ap.add_argument("--use-learned-mixer", action="store_true",
                    help="enable the cmix-style learned gating mixer; tiny "
                         "MLP outputs per-step weights from per-submodel "
                         "confidence features. equal-weight ensemble is in "
                         "the mixer's hypothesis class so it can't degrade "
                         "below it at convergence.")
    ap.add_argument("--mixer-lr", type=float, default=0.01,
                    help="LR for the learned mixer's Adam optimizer "
                         "(separate from submodel LR schedules).")
    ap.add_argument("--clip-xl", type=float, default=None,
                    help="grad-norm clip for the Transformer-XL path "
                         "(default 0.25, NNCP-aligned).")
    ap.add_argument("--adam-eps-xl", type=float, default=None,
                    help="Adam epsilon for the Transformer-XL path "
                         "(default 1e-9, NNCP-aligned).")
    args = ap.parse_args()

    prepared = args.output_nb + ".prepared.ipynb"
    prepare_notebook(SRC_NB, prepared)

    params = {
        "path_to_file": args.input,
        "preprocess": args.preprocess,
        "use_bf16": args.use_bf16,
        "use_fp16": args.use_fp16,
        "use_deterministic": args.use_deterministic,
        "mode": args.mode,
        "tensorboard": True,
        "tensorboard_run_name": args.tb_run_name,
        "tensorboard_logdir": args.tb_logdir,
        "download_option": "no_download",
        "checkpoint": False,
        "total_parts": 1,
        "current_part": 1,
        "local_upload": False,
        "http_path": "",
        "custom_path": "",
        # Transformer-XL routing + hparams. Read only when model_type matches;
        # otherwise harmless globals that the LSTM path ignores.
        "model_type": args.model_type,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "d_model": args.d_model,
        "d_head": args.d_head,
        "d_inner": args.d_inner,
        "mem_len": args.mem_len,
        "ext_tgt_len": args.ext_tgt_len,
        "attn_type": args.attn_type,
        "tied_r_bias": bool(args.tied_r_bias),
        "use_gelu": bool(args.use_gelu),
        "dropout": args.dropout,
        "dropatt": args.dropatt,
        "init_std": args.init_std,
        "retrain_tgt_len": args.retrain_tgt_len,
        "retrain_mem_len": args.retrain_mem_len,
        "learning_rate_schedule_xl": args.xl_lr_schedule,
        "retrain_lr_schedule_xl": args.xl_retrain_lr_schedule,
    }
    # Data-pipeline hparams: only inject when explicitly set, so the notebook
    # default applies otherwise. Avoids silently changing behaviour for
    # existing run scripts that don't pass these new flags.
    if args.batch_size is not None: params["batch_size"] = args.batch_size
    if args.n_words is not None: params["n_words"] = args.n_words
    if args.min_freq is not None: params["min_freq"] = args.min_freq
    if args.retrain_block_len is not None: params["retrain_block_len"] = args.retrain_block_len
    if args.retrain_batch_size is not None: params["retrain_batch_size"] = args.retrain_batch_size
    if args.retrain_period_schedule is not None: params["retrain_period_schedule"] = args.retrain_period_schedule
    # Hybrid learned mixer (no-op unless model_type=="hybrid")
    params["use_learned_mixer"] = bool(args.use_learned_mixer)
    params["mixer_lr"] = args.mixer_lr
    if args.clip_xl is not None: params["clip_xl"] = args.clip_xl
    if args.adam_eps_xl is not None: params["adam_eps_xl"] = args.adam_eps_xl

    print(f"[run_papermill] input={args.input} use_bf16={args.use_bf16} use_fp16={args.use_fp16} preprocess={args.preprocess}")
    print(f"[run_papermill] tb_run_name={args.tb_run_name}")
    print(f"[run_papermill] params: {json.dumps(params, indent=2)}")

    t0 = time.time()
    pm.execute_notebook(
        prepared,
        args.output_nb,
        parameters=params,
        cwd=REPO,
        log_output=True,
        progress_bar=False,
    )
    elapsed = time.time() - t0

    compressed_path = os.path.join(REPO, "data", "compressed.dat")
    compressed_size = os.path.getsize(compressed_path) if os.path.exists(compressed_path) else None
    input_size = os.path.getsize(os.path.join(REPO, args.input))
    bpc = (compressed_size * 8) / input_size if compressed_size else None

    print()
    print("=" * 72)
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"input bytes:      {input_size:,}")
    print(f"compressed bytes: {compressed_size:,}")
    print(f"bpc:              {bpc:.4f}")
    print("=" * 72)

    if args.save_compressed and compressed_size:
        shutil.copy2(compressed_path, args.save_compressed)
        print(f"saved compressed artifact -> {args.save_compressed}")


if __name__ == "__main__":
    main()
