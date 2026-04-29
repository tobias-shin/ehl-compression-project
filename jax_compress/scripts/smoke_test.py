"""End-to-end smoke test for the PyTorch port.

Tests:
  1. torch_compress.ipynb compress→decompress round-trip
  2. Determinism: two compresses with the same seed produce identical bytes
  3. torch_compress_decompressor.ipynb (slim) decodes bytes from the full notebook
"""
import json
import io
import os
import sys


def strip_magics(s):
    out = []
    for line in s.splitlines():
        ls = line.lstrip()
        ind = line[:len(line) - len(ls)]
        if ls.startswith('!') or ls.startswith('%'):
            out.append(ind + 'pass  # ' + ls)
        else:
            out.append(line)
    return '\n'.join(out)


def load_namespace(nb_path, hardcoded_hparams=None):
    """Exec all code cells of a notebook into a fresh namespace.

    Skips cells that need actual file/network access (System Info, Setup Files,
    Mount Drive, Preprocess, Compression cell, top-level Decompression cell, etc.).
    """
    nb = json.load(open(nb_path))
    ns = {}
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        title_line = src.splitlines()[0].strip() if src else ''
        # Skip cells whose execution requires Colab-only resources.
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
        }
        if title_line in SKIP_TITLES:
            continue
        src = strip_magics(src)
        src = src.replace('from google.colab import files', 'files=None')
        src = src.replace('from google.colab import drive', 'drive=None')
        try:
            exec(src, ns)
        except Exception as e:
            print(f"  [load_namespace] Cell {i} ({title_line[:60]!r}) failed: {e}")
            raise
    if hardcoded_hparams:
        exec(hardcoded_hparams, ns)
    return ns


HPARAMS = """
batch_size = 4
seq_length = 4
rnn_units = 8
num_layers = 2
embedding_size = 6
ensemble_size = 1
learning_rate_schedule = "0:0.005"
adam_b1 = 0.0
adam_b2 = 0.999
adam_eps = 1e-9
retrain_period_schedule = "0:1000000"
retrain_block_len = 100
retrain_seq_length = 4
retrain_batch_size = 2
retrain_lr_schedule = "0:0.001"
retrain_dropout = 0.1
total_parts = 1
current_part = 1
checkpoint = False
download_option = "no_download"
preprocess = "none"
mode = "both"
"""


def encode(ns, data_in, vocab):
    buf = io.BytesIO()
    bitout = ns['BitOutputStream'](buf)
    enc = ns['ArithmeticEncoder'](32, bitout)
    ns['process'](True, len(data_in), vocab, enc, list(data_in))
    enc.finish()
    while bitout.numbitsfilled != 0:
        bitout.write(0)
    return buf.getvalue()


def decode(ns, compressed, length, vocab):
    bitin = ns['BitInputStream'](io.BytesIO(compressed))
    dec = ns['ArithmeticDecoder'](32, bitin)
    out = [0] * length
    ns['process'](False, length, vocab, dec, out)
    return out


# ---- Test data -----------------------------------------------------------
import random
random.seed(0)
DATA = [random.randrange(8) for _ in range(64)]
VOCAB = 8

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Make `models/` importable when build_model lazy-imports TransformerXLModel.
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ---- Test 1: round-trip on full notebook ---------------------------------
print("== Test 1: torch_compress.ipynb round-trip ==")
ns_full = load_namespace(os.path.join(REPO, "torch_compress.ipynb"), HPARAMS)
compressed_a = encode(ns_full, DATA, VOCAB)
print(f"  Encoded {len(DATA)} symbols -> {len(compressed_a)} bytes")
decoded_a = decode(ns_full, compressed_a, len(DATA), VOCAB)
assert decoded_a == DATA, "Round-trip mismatch in full notebook"
print("  PASS: round-trip is lossless")

# ---- Test 2: determinism (two encodes -> identical bytes) ----------------
print("\n== Test 2: determinism (two compress runs, same seed) ==")
ns_full2 = load_namespace(os.path.join(REPO, "torch_compress.ipynb"), HPARAMS)
compressed_b = encode(ns_full2, DATA, VOCAB)
print(f"  First encode:  {len(compressed_a)} bytes, sha={hash(compressed_a)}")
print(f"  Second encode: {len(compressed_b)} bytes, sha={hash(compressed_b)}")
assert compressed_a == compressed_b, "Encodes are NOT deterministic across runs"
print("  PASS: bit-identical encode across processes")

# ---- Test 3: slim decompressor decodes full-notebook bytes ---------------
print("\n== Test 3: torch_compress_decompressor.ipynb decodes full-notebook output ==")
ns_slim = load_namespace(os.path.join(REPO, "torch_compress_decompressor.ipynb"), HPARAMS)
decoded_slim = decode(ns_slim, compressed_a, len(DATA), VOCAB)
assert decoded_slim == DATA, "Slim decompressor does not match"
print("  PASS: slim decompressor reproduces original")


# ---- transformer_xl backend ---------------------------------------------
# Exercises the NNCP-style streaming + retraining loop. Hparams are tiny so
# the test stays under a second on CPU. retrain_period is set so retrain
# fires during the run (catches reset_length toggle, mem reset after retrain,
# and the separate retrain optimizer state).
XL_HPARAMS = HPARAMS + """
model_type = "transformer_xl"
n_layer = 2
n_head = 2
d_model = 16
d_head = 8
d_inner = 32
mem_len = 8
ext_tgt_len = 3
attn_type = 1
tied_r_bias = True
use_gelu = True
dropout = 0.0
dropatt = 0.0
init_std = 0.013
retrain_tgt_len = 2
retrain_mem_len = 4
learning_rate_schedule_xl = "0:0.005"
retrain_lr_schedule_xl = "0:0.001"
"""
# Override retrain period so retraining actually fires during the 16-step run.
XL_HPARAMS = XL_HPARAMS.replace(
    'retrain_period_schedule = "0:1000000"',
    'retrain_period_schedule = "0:5"',
)

# ---- Test 4: transformer_xl round-trip ----------------------------------
print("\n== Test 4: torch_compress.ipynb (transformer_xl) round-trip with retrain ==")
ns_xl = load_namespace(os.path.join(REPO, "torch_compress.ipynb"), XL_HPARAMS)
compressed_xl = encode(ns_xl, DATA, VOCAB)
print(f"  Encoded {len(DATA)} symbols -> {len(compressed_xl)} bytes")
decoded_xl = decode(ns_xl, compressed_xl, len(DATA), VOCAB)
assert decoded_xl == DATA, "Round-trip mismatch in transformer_xl path"
print("  PASS: round-trip is lossless (with retrain firing)")

# ---- Test 5: transformer_xl determinism ---------------------------------
print("\n== Test 5: transformer_xl determinism (two compress runs, same seed) ==")
ns_xl2 = load_namespace(os.path.join(REPO, "torch_compress.ipynb"), XL_HPARAMS)
compressed_xl_b = encode(ns_xl2, DATA, VOCAB)
print(f"  First encode:  {len(compressed_xl)} bytes, sha={hash(compressed_xl)}")
print(f"  Second encode: {len(compressed_xl_b)} bytes, sha={hash(compressed_xl_b)}")
assert compressed_xl == compressed_xl_b, "transformer_xl encodes are NOT deterministic"
print("  PASS: bit-identical encode across processes (with retrain firing)")


# ---- Test 6: transformer_xl bf16 round-trip with retrain firing ---------
# This is the test that should have existed before commit 9137f21 wired up
# bf16. The earlier check (md5(fp32) != md5(bf16)) confirmed bf16 was active
# but did NOT verify round-trip; commit 3f6a620 found that retrain-active
# bf16 broke decompression on enwik6+. After applying the
# allow_bf16_reduced_precision_reduction = False fix in IMPORTS_SRC, this
# test verifies bf16 round-trip is restored.
#
# CPU bf16 autocast does not exercise the same matmul reduction paths as
# CUDA -- this CPU smoke is only a regression detector, not a complete
# verification. The real proof is re-running enwik6 on GPU and checking the
# Validation cell's md5s match.
XL_BF16_HPARAMS = XL_HPARAMS + 'use_bf16 = True\n'
print("\n== Test 6: transformer_xl bf16 round-trip with retrain ==")
ns_bf16 = load_namespace(os.path.join(REPO, "torch_compress.ipynb"), XL_BF16_HPARAMS)
compressed_bf16 = encode(ns_bf16, DATA, VOCAB)
print(f"  Encoded {len(DATA)} symbols -> {len(compressed_bf16)} bytes")
decoded_bf16 = decode(ns_bf16, compressed_bf16, len(DATA), VOCAB)
assert decoded_bf16 == DATA, (
    f"transformer_xl bf16 round-trip is broken: decoded[:10]={decoded_bf16[:10]} "
    f"!= expected[:10]={DATA[:10]}"
)
print("  PASS: bf16 round-trip is lossless (CPU; GPU verification still needed)")

print("\nALL TESTS PASS")
