"""Local CLI driver for torch_compress.ipynb.

Lets you compress / decompress a real file outside Colab, using the same
model and arithmetic-coder as the notebook. Uses byte-level vocabulary
(preprocess='none') so it does not depend on NNCP / cmix being installed.

Usage:
  # Compress
  python scripts/run_local.py compress   <input>  <compressed.bin>
  # Decompress
  python scripts/run_local.py decompress <compressed.bin> <output>
  # Round-trip self-check on a sample file
  python scripts/run_local.py roundtrip  <input>

Defaults are sized for a small CPU run (rnn_units=64, num_layers=2). Override
with --rnn-units / --num-layers / --batch-size / --seq-length / --embedding-size
to match the full Colab config (rnn_units=1400, num_layers=8) for a real run.
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys


# --------------------------------------------------------------------------
# Notebook code loader
# --------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(REPO, "torch_compress.ipynb")

# Make the repo's `models/` package importable when build_model lazy-imports
# TransformerXLModel. From scripts/, sys.path[0] is the script's directory and
# does not include the repo root, so `from models.transformer_xl import ...`
# would fail without this.
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SKIP_TITLES = {
    '#@title System Info',
    '#@title Mount Google Drive',
    '#@title Setup Files',
    '#@title Preprocess',
    '#@title Compression',
    '#@title Decompression',
    '#@title Download Result',
    '#@title Validation',
    '#@title Close Runtime',
    '#@title Verify',
    '#@title Parameters',
}


def _strip_magics(s):
    out = []
    for line in s.splitlines():
        ls = line.lstrip()
        ind = line[: len(line) - len(ls)]
        if ls.startswith('!') or ls.startswith('%'):
            out.append(ind + 'pass  # ' + ls)
        else:
            out.append(line)
    return '\n'.join(out)


def load_notebook_namespace(hparams_src):
    """Exec the notebook's code cells (skipping Colab-only ones) into a fresh ns."""
    nb = json.load(open(NB_PATH))
    ns = {}
    # Hyperparameters first — the Compression Library cell references them at module level.
    exec(hparams_src, ns)
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        title = src.splitlines()[0].strip() if src else ''
        if title in SKIP_TITLES:
            continue
        src = _strip_magics(src)
        src = src.replace('from google.colab import files', 'files=None')
        src = src.replace('from google.colab import drive', 'drive=None')
        exec(src, ns)
    return ns


# --------------------------------------------------------------------------
# File I/O wrapper (header, vocab bitmap, arithmetic-coder framing)
# --------------------------------------------------------------------------

# Layout of compressed file (preprocess='none', byte-level):
#   bytes  0..4       : 5-byte big-endian length of the original file
#   bits   40..40+255 : 256-bit vocabulary bitmap (which bytes appear in the input)
#   bits   thereafter : arithmetic-coded payload


def compress_file(ns, in_path, out_path):
    text = open(in_path, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    char2idx = {c: i for i, c in enumerate(vocab)}
    int_list = [char2idx[b] for b in text]
    # Round vocab up to multiple of 8 (matches notebook).
    rounded_vocab = math.ceil(vocab_size / 8) * 8
    length = len(int_list)

    print(f"input bytes={length}  unique bytes={vocab_size}  rounded vocab={rounded_vocab}")

    with open(out_path, 'wb') as out:
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        bitout = ns['BitOutputStream'](out)
        enc = ns['ArithmeticEncoder'](32, bitout)
        # 256-bit bitmap of which bytes appear in input.
        present = set(vocab)
        for i in range(256):
            bitout.write(1 if i in present else 0)
        ns['process'](True, length, rounded_vocab, enc, int_list)
        enc.finish()
        bitout.close()

    print(f"wrote {out_path}: {os.path.getsize(out_path)} bytes")


def decompress_file(ns, in_path, out_path):
    with open(in_path, 'rb') as inp:
        header = inp.read(5)
        length = int.from_bytes(header, byteorder='big')
        bitin = ns['BitInputStream'](inp)
        vocab = [i for i in range(256) if bitin.read()]
        rounded_vocab = math.ceil(len(vocab) / 8) * 8

        print(f"output bytes={length}  unique bytes={len(vocab)}  rounded vocab={rounded_vocab}")

        dec = ns['ArithmeticDecoder'](32, bitin)
        output = [0] * length
        ns['process'](False, length, rounded_vocab, dec, output)

    idx2char = bytes(vocab)
    with open(out_path, 'wb') as f:
        f.write(bytes(idx2char[i] for i in output))

    print(f"wrote {out_path}: {os.path.getsize(out_path)} bytes")


# --------------------------------------------------------------------------
# Hyperparameters template
# --------------------------------------------------------------------------

def build_hparams(args):
    return f"""
batch_size = {args.batch_size}
seq_length = {args.seq_length}
rnn_units = {args.rnn_units}
num_layers = {args.num_layers}
embedding_size = {args.embedding_size}
ensemble_size = 1
learning_rate_schedule = "{args.lr_schedule}"
adam_b1 = 0.0
adam_b2 = 0.9999
adam_eps = 1e-12
retrain_period_schedule = "{args.retrain_period}"
retrain_block_len = {args.retrain_block_len}
retrain_seq_length = {args.retrain_seq_length}
retrain_batch_size = {args.retrain_batch_size}
retrain_lr_schedule = "{args.retrain_lr_schedule}"
retrain_dropout = {args.retrain_dropout}
total_parts = 1
current_part = 1
checkpoint = False
download_option = "no_download"
preprocess = "none"
mode = "{args.mode}"
tensorboard = {bool(args.tensorboard)}
tensorboard_run_name = "{args.tb_run_name}"
tensorboard_logdir = "{args.tb_logdir}"
model_type = "{args.model_type}"
n_layer = {args.n_layer}
n_head = {args.n_head}
d_model = {args.d_model}
d_head = {args.d_head}
d_inner = {args.d_inner}
mem_len = {args.mem_len}
ext_tgt_len = {args.ext_tgt_len}
attn_type = {args.attn_type}
tied_r_bias = {bool(args.tied_r_bias)}
use_gelu = {bool(args.use_gelu)}
dropout = {args.dropout}
dropatt = {args.dropatt}
init_std = {args.init_std}
retrain_tgt_len = {args.retrain_tgt_len}
retrain_mem_len = {args.retrain_mem_len}
learning_rate_schedule_xl = "{args.xl_lr_schedule}"
retrain_lr_schedule_xl = "{args.xl_retrain_lr_schedule}"
use_bf16 = {bool(args.use_bf16)}
"""


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Local compress/decompress driver for torch_compress.ipynb")
    ap.add_argument('action', choices=['compress', 'decompress', 'roundtrip'])
    ap.add_argument('input')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seq-length', type=int, default=10)
    ap.add_argument('--rnn-units', type=int, default=64)
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--embedding-size', type=int, default=32)
    ap.add_argument('--lr-schedule', default="0:0.001")
    # Retrain (NNCP-style periodic re-fit on recent history). Enabled by default;
    # use --no-retrain to disable for very short runs where it just adds overhead.
    ap.add_argument('--retrain-period', default="0:1001 200000:5001",
                    help='retrain period schedule; "0:1001" = retrain every 1001 steps')
    ap.add_argument('--retrain-block-len', type=int, default=100000,
                    help='number of trailing symbols included in each retrain pass')
    ap.add_argument('--retrain-seq-length', type=int, default=100,
                    help='BPTT sequence length used during retrain')
    ap.add_argument('--retrain-batch-size', type=int, default=64,
                    help='batch size for retrain (smaller than the Colab 256 default for CPU friendliness)')
    ap.add_argument('--retrain-lr-schedule', default="0:0.0005 200000:0.0002")
    ap.add_argument('--retrain-dropout', type=float, default=0.4)
    ap.add_argument('--no-retrain', action='store_true',
                    help='disable periodic retraining entirely (overrides --retrain-period)')
    ap.add_argument('--tensorboard', action='store_true',
                    help="enable tensorboard logging (writes to <tb-logdir>/<tb-run-name>)")
    ap.add_argument('--tb-logdir', default='data/tensorboard',
                    help="parent directory for tensorboard event files")
    ap.add_argument('--tb-run-name', default='torch_local',
                    help="subdirectory name for this run; use distinct names to compare in tensorboard")
    # ---- Transformer-XL --------------------------------------------------
    # Default model is the LSTM. Pass --model-type transformer_xl to route to
    # the NNCP-style streaming + retraining loop. The rest of the --xl-* flags
    # default to NNCP_v2 nncp_enwik_base.sh values; they are read only when
    # model_type == "transformer_xl".
    ap.add_argument('--model-type', choices=['lstm', 'transformer_xl', 'hybrid'], default='lstm')
    ap.add_argument('--n-layer', type=int, default=12)
    ap.add_argument('--n-head', type=int, default=8)
    ap.add_argument('--d-model', type=int, default=512)
    ap.add_argument('--d-head', type=int, default=64)
    ap.add_argument('--d-inner', type=int, default=2048)
    ap.add_argument('--mem-len', type=int, default=160)
    ap.add_argument('--ext-tgt-len', type=int, default=31)
    ap.add_argument('--attn-type', type=int, default=1)
    ap.add_argument('--tied-r-bias', type=int, default=1, help='1 = tied, 0 = per-layer')
    ap.add_argument('--use-gelu', type=int, default=1, help='1 = GELU, 0 = ReLU')
    ap.add_argument('--dropout', type=float, default=0.25)
    ap.add_argument('--dropatt', type=float, default=0.0)
    ap.add_argument('--init-std', type=float, default=0.013)
    ap.add_argument('--retrain-tgt-len', type=int, default=64)
    ap.add_argument('--retrain-mem-len', type=int, default=128)
    ap.add_argument('--xl-lr-schedule',
                    default="0:7.9e-5 341105:1.6e-5 3134681:5.0e-6",
                    help="streaming LR schedule used when model_type == transformer_xl")
    ap.add_argument('--xl-retrain-lr-schedule',
                    default="0:4.0e-4 13000:2.0e-4 93000:1.0e-4 163000:5.0e-5 1911300:1.6e-5",
                    help="retrain LR schedule used when model_type == transformer_xl")
    # ---- Mixed precision -------------------------------------------------
    # Default fp32 (matches run_papermill.py default) -- bf16 mixed precision
    # is round-trip safe on the transformer path post-commit 7611a88, but at
    # the file sizes typically driven from this CLI it's slightly slower than
    # fp32. Set --use-bf16 to opt in.
    ap.add_argument('--use-bf16', dest='use_bf16', action='store_true')
    ap.add_argument('--no-bf16', dest='use_bf16', action='store_false')
    ap.set_defaults(use_bf16=False)
    args = ap.parse_args()
    if args.no_retrain:
        args.retrain_period = "0:1000000000"

    # Mode goes into hparams so the notebook code branches correctly.
    args.mode = 'compress' if args.action == 'compress' else 'decompress'
    if args.action == 'roundtrip':
        args.mode = 'both'

    hparams = build_hparams(args)
    ns = load_notebook_namespace(hparams)

    if args.action == 'compress':
        if not args.output:
            ap.error("compress requires <output>")
        compress_file(ns, args.input, args.output)

    elif args.action == 'decompress':
        if not args.output:
            ap.error("decompress requires <output>")
        decompress_file(ns, args.input, args.output)

    else:  # roundtrip
        comp_path = args.input + ".zc"
        dec_path = args.input + ".dec"
        print(f"[1/2] compressing {args.input} -> {comp_path}")
        compress_file(ns, args.input, comp_path)
        # Reload so internal seeds reset; mirrors what a separate decode invocation does.
        ns2 = load_notebook_namespace(hparams.replace('mode = "both"', 'mode = "decompress"'))
        print(f"[2/2] decompressing {comp_path} -> {dec_path}")
        decompress_file(ns2, comp_path, dec_path)
        m_in = md5(args.input)
        m_out = md5(dec_path)
        ratio = os.path.getsize(comp_path) / os.path.getsize(args.input)
        print(f"\nmd5(input)  = {m_in}")
        print(f"md5(output) = {m_out}")
        print(f"compression ratio = {ratio:.3f} ({os.path.getsize(comp_path)} / {os.path.getsize(args.input)})")
        if m_in != m_out:
            print("FAIL: md5 mismatch")
            sys.exit(1)
        print("PASS: lossless round-trip")


if __name__ == '__main__':
    main()
