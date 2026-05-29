"""Build the PyTorch port notebooks from the original JAX notebook.

Reads jax_compress.ipynb, replaces the JAX/Flax/Optax-specific cells with
PyTorch equivalents, and writes:
  * torch_compress.ipynb              -- full notebook (compress + decompress)
  * torch_compress_decompressor.ipynb -- slim LTCB-style decompressor

Run from anywhere: `python scripts/build_torch_notebooks.py`.
"""

import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(REPO_DIR, "jax_compress.ipynb")
DST_NB = os.path.join(REPO_DIR, "torch_compress.ipynb")


def code_cell(source: str, tags=None):
    metadata = {}
    if tags:
        metadata["tags"] = list(tags)
    return {
        "cell_type": "code",
        "metadata": metadata,
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell sources (PyTorch port)
# ---------------------------------------------------------------------------

IMPORTS_SRC = '''#@title Imports

# --- LTCB determinism preamble ---------------------------------------------
# CUBLAS_WORKSPACE_CONFIG must be set before any CUDA initialization, otherwise
# torch.use_deterministic_algorithms will refuse to enable strict mode.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Force bf16 matmul accumulation in fp32 (default is bf16 reduction). Without
# this, parallel reductions in bf16 matmul kernels on CUDA produce bit-
# different outputs across kernel launches given identical inputs, which
# breaks the encode/decode probability agreement the arithmetic coder needs.
# No-op on fp32 runs; ~10-20% slowdown vs bf16-reduce on bf16 runs in
# exchange for round-trip-safe determinism.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
# ---------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
try:
  from google.colab import files
except ImportError:
  files = None
import time
import math
import sys
import subprocess
import contextlib
from typing import Any, List, Tuple
try:
  from google.colab import drive
except ImportError:
  drive = None
import pickle
'''


SYSTEM_INFO_SRC = '''#@title System Info

def system_info():
  """Prints out system information."""
  gpu_info = !nvidia-smi
  gpu_info = '\\n'.join(gpu_info)
  if gpu_info.find('failed') >= 0:
    print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
    print('and then re-execute this cell.')
  else:
    print(gpu_info)
  print("PyTorch version:", torch.__version__)
  print("CUDA available:", torch.cuda.is_available())
  if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
  !lscpu |grep 'Model name'
  !cat /proc/meminfo | head -n 3

system_info()
'''


MODEL_ARCH_SRC = '''#@title Model Architecture

class LSTMModel(nn.Module):
  """LSTM stack with skip connections from the embedding into every layer's input.

  Mirrors the Flax model: each layer (after the first) sees concat(embedding, prev_output);
  the dense head sees concat of all layers' outputs.

  Implementation note: uses nn.LSTMCell in a manual time loop instead of nn.LSTM.
  cuDNN's RNN kernel is non-deterministic even with cudnn.deterministic=True, which
  would break the LTCB compress/decompress invariant. The cell-loop is slower but
  produces bit-identical predictions across runs on the same machine.
  """

  def __init__(self, vocab_size, embedding_size, rnn_units, num_layers, dropout_rate=0.0):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.rnn_units = rnn_units
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate

    self.embed = nn.Embedding(vocab_size, embedding_size)

    self.lstm_cells = nn.ModuleList()
    for i in range(num_layers):
      input_size = embedding_size if i == 0 else embedding_size + rnn_units
      self.lstm_cells.append(nn.LSTMCell(input_size, rnn_units))

    out_size = rnn_units * num_layers if num_layers > 1 else rnn_units
    self.dense_logits = nn.Linear(out_size, vocab_size)

  def init_states(self, batch_size, device):
    """Returns a list of (h, c) tuples (one per layer), each (batch, rnn_units)."""
    return [
        (torch.zeros(batch_size, self.rnn_units, device=device),
         torch.zeros(batch_size, self.rnn_units, device=device))
        for _ in range(self.num_layers)
    ]

  def forward(self, inputs, states, return_sequence=False, deterministic=True):
    """
    Args:
      inputs: (batch, seq_length) int64
      states: list of (h, c) tuples per layer; each (batch, rnn_units)
      return_sequence: if True, returns logits for the entire sequence
      deterministic: if True, dropout is disabled (matches Flax's deterministic=True)
    Returns:
      logits: (batch, vocab) if not return_sequence else (batch, seq, vocab)
      new_states: list of (h, c) tuples per layer
    """
    use_dropout = not deterministic
    embedding = F.dropout(self.embed(inputs), p=self.dropout_rate, training=use_dropout)
    seq_len = embedding.shape[1]

    new_states = list(states)
    layer_outputs = []
    curr_input = embedding

    for i, cell in enumerate(self.lstm_cells):
      h, c = states[i]
      step_outputs = []
      for t in range(seq_len):
        h, c = cell(curr_input[:, t, :], (h, c))
        step_outputs.append(h)
      layer_out = torch.stack(step_outputs, dim=1)  # (batch, seq, rnn_units)
      new_states[i] = (h, c)

      if i < self.num_layers - 1:
        layer_out = F.dropout(layer_out, p=self.dropout_rate, training=use_dropout)
      layer_outputs.append(layer_out)

      if i < self.num_layers - 1:
        curr_input = torch.cat([embedding, layer_out], dim=-1)

    if return_sequence:
      final_rep = layer_outputs[0] if self.num_layers == 1 else torch.cat(layer_outputs, dim=-1)
      logits = self.dense_logits(final_rep)
    else:
      if self.num_layers == 1:
        final_rep = layer_outputs[0][:, -1, :]
      else:
        final_rep = torch.cat([o[:, -1, :] for o in layer_outputs], dim=-1)
      logits = self.dense_logits(final_rep)

    return logits.float(), new_states


def build_model(model_type, *, vocab_size, embedding_size, rnn_units, num_layers,
                dropout_rate=0.0):
  """Construct the predictor selected by ``model_type``.

  The default "lstm" returns a ``LSTMModel`` constructed identically to before,
  preserving the existing reference behaviour. "transformer_xl" lazy-imports
  the local NNCP-style adapter; it is unavailable in a vanilla Colab session
  that has not cloned this repo.

  Transformer hparams are read from notebook globals (``n_layer``, ``n_head``,
  ``d_model``, ``d_head``, ``d_inner``, ``mem_len``, ``ext_tgt_len``,
  ``attn_type``, ``tied_r_bias``, ``use_gelu``, ``dropout``, ``dropatt``,
  ``init_std``) so the LSTM call sites do not need to change. Missing globals
  fall back to the adapter's NNCP-base defaults.
  """
  if model_type == "lstm":
    return LSTMModel(vocab_size=vocab_size, embedding_size=embedding_size,
                     rnn_units=rnn_units, num_layers=num_layers,
                     dropout_rate=dropout_rate)
  if model_type == "transformer_xl":
    from models.transformer_xl import TransformerXLModel
    g = globals()
    return TransformerXLModel(
        vocab_size=vocab_size,
        n_layer=g.get('n_layer', 12),
        n_head=g.get('n_head', 8),
        d_model=g.get('d_model', 512),
        d_head=g.get('d_head', 64),
        d_inner=g.get('d_inner', 2048),
        dropout=g.get('dropout', 0.0),
        dropatt=g.get('dropatt', 0.0),
        tgt_len=1,  # streaming inference shape; reset_length toggles for retrain
        ext_len=g.get('ext_tgt_len', 31),
        mem_len=g.get('mem_len', 160),
        attn_type=g.get('attn_type', 1),
        tied_r_bias=g.get('tied_r_bias', True),
        use_gelu=g.get('use_gelu', True),
        init_std=g.get('init_std', 0.02),
    )
  if model_type == "hybrid":
    # Construct LSTM + Transformer-XL via the same code paths used for the
    # solo cases, then wrap them in HybridModel. Both submodels are unmodified;
    # the hybrid is purely composition.
    lstm = LSTMModel(vocab_size=vocab_size, embedding_size=embedding_size,
                     rnn_units=rnn_units, num_layers=num_layers,
                     dropout_rate=dropout_rate)
    from models.transformer_xl import TransformerXLModel
    g = globals()
    xl = TransformerXLModel(
        vocab_size=vocab_size,
        n_layer=g.get('n_layer', 12),
        n_head=g.get('n_head', 8),
        d_model=g.get('d_model', 512),
        d_head=g.get('d_head', 64),
        d_inner=g.get('d_inner', 2048),
        dropout=g.get('dropout', 0.0),
        dropatt=g.get('dropatt', 0.0),
        tgt_len=1,
        ext_len=g.get('ext_tgt_len', 31),
        mem_len=g.get('mem_len', 160),
        attn_type=g.get('attn_type', 1),
        tied_r_bias=g.get('tied_r_bias', True),
        use_gelu=g.get('use_gelu', True),
        init_std=g.get('init_std', 0.02),
    )
    from models.hybrid import HybridModel
    # Optional learned mixer (cmix-style context-dependent gating). When the
    # `use_learned_mixer` notebook global is True, attach a tiny MLP that
    # produces per-step weights from per-submodel confidence features
    # (entropy + max log-prob). When False, HybridModel.mixer is None and
    # the ensemble loop falls back to equal-weight geometric mean.
    mixer = None
    if g.get('use_learned_mixer', False):
      from models.learned_mixer import LearnedMixer
      mixer = LearnedMixer(
          n_models=2,
          hidden_dim=g.get('mixer_hidden_dim', 64),
          init_std=g.get('mixer_init_std', 0.02),
      )
    return HybridModel(lstm, xl, mixer=mixer)
  raise ValueError(f"unknown model_type: {model_type!r}")
'''


COMPRESSION_LIB_SRC = '''#@title Compression Library


def parse_schedule(schedule_str):
  """Parses a learning rate or training schedule string.

  Expects a string formatted as "step:value step:value" and
  returns a list of parsed (step, value) tuples sorted by step.
  """
  points = []
  try:
    for item in schedule_str.split():
      step, value = item.split(':')
      points.append((float(step), float(value)))
    points.sort(key=lambda x: x[0])
  except ValueError:
    print(f"Error parsing schedule: {schedule_str}")
    return []
  return points


def get_scheduled_value(schedule, step):
  """Linearly interpolates a value at the given step using the schedule.

  Outside the bounded range, the closest endpoint value is returned.
  """
  if not schedule:
    return 0.0
  if step <= schedule[0][0]:
    return schedule[0][1]
  if step >= schedule[-1][0]:
    return schedule[-1][1]
  for i in range(len(schedule) - 1):
    start_step, start_val = schedule[i]
    end_step, end_val = schedule[i+1]
    if start_step <= step <= end_step:
      fraction = (step - start_step) / (end_step - start_step)
      return start_val + fraction * (end_val - start_val)
  return schedule[-1][1]


def get_symbol(index, length, freq, coder, compress, data):
  """Reads or writes a single symbol via the arithmetic coder."""
  symbol = 0
  if index < length:
    if compress:
      symbol = data[index]
      coder.write(freq, symbol)
    else:
      symbol = coder.read(freq)
      data[index] = symbol
  return symbol


def reset_seed():
  SEED = 1234
  os.environ['PYTHONHASHSEED'] = str(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def download(path):
  if download_option == 'local':
    files.download(path)
  elif download_option == 'google_drive':
    !cp -f $path /content/gdrive/My\\ Drive


def detach_states(states_per_model):
  """Detach a list (over ensemble) of lists (over layers) of (h, c) tuples."""
  return [
      [(h.detach(), c.detach()) for (h, c) in layer_states]
      for layer_states in states_per_model
  ]


def _resolve_precision():
  """Pick the mixed-precision dtype from globals. Priority: fp16 > bf16 > fp32.

  Returns (ac_dtype, enabled). When CUDA is unavailable, falls back to fp32
  regardless of the flags so smoke tests on CPU still work.
  """
  if not torch.cuda.is_available():
    return torch.float32, False
  if bool(globals().get('use_fp16', False)):
    return torch.float16, True
  if bool(globals().get('use_bf16', False)):
    return torch.bfloat16, True
  return torch.float32, False


def _lstm_autocast(device):
  """Returns a mixed-precision autocast context per ``use_fp16``/``use_bf16``.

  When disabled, returns a no-op autocast context so call sites stay uniform.
  Parameters and the loss/backward stay in fp32; only the forward matmuls inside
  ``LSTMModel`` run in the autocast dtype. The model's ``return logits.float()``
  upcasts the logits at the boundary so the arithmetic coder always sees fp32
  probabilities.
  """
  ac_dtype, enabled = _resolve_precision()
  return torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=enabled)


def forward_ensemble(models, inputs, states_per_model):
  """Runs forward pass for each ensemble member with autograd enabled.

  Returns:
    probs_np: numpy array (batch, vocab), geometric-mean ensemble probabilities
    logits_list: list of (batch, vocab) tensors with grad
    new_states_per_model: list (over ensemble) of lists of (h, c)
  """
  log_probs_list = []
  logits_list = []
  new_states_per_model = []
  for model, states in zip(models, states_per_model):
    # deterministic=True matches the JAX path used for online updates (no dropout).
    # We deliberately do NOT call model.eval(): cuDNN LSTM requires train() mode for backward.
    with _lstm_autocast(inputs.device):
      logits, new_states = model(inputs, states, return_sequence=False, deterministic=True)
    logits_list.append(logits)
    log_probs_list.append(F.log_softmax(logits, dim=-1))
    new_states_per_model.append(new_states)
  mean_log_probs = torch.stack(log_probs_list, dim=0).mean(dim=0)
  probs = F.softmax(mean_log_probs, dim=-1).detach().cpu().numpy()
  return probs, logits_list, new_states_per_model


def backward_and_step(logits_list, optimizers, models, symbols, mask, vocab_size,
                      clip=4.0):
  """Cross-entropy backward + gradient clip + optimizer step for every ensemble member.

  Returns the geometric-mean ensemble loss (scalar) and the mask sum (denominator).

  ``clip`` defaults to 4.0 (torch_compress's tuned value for the LSTM); the
  Transformer-XL streaming path passes clip=clip_xl (NNCP-aligned 0.25).
  """
  log_probs_list = []
  for logits, optimizer, model in zip(logits_list, optimizers, models):
    optimizer.zero_grad()
    loss_per = F.cross_entropy(logits, symbols, reduction='none')
    loss = (loss_per * mask).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    log_probs_list.append(F.log_softmax(logits.detach(), dim=-1))

  mean_log_probs = torch.stack(log_probs_list, dim=0).mean(dim=0)
  one_hot = F.one_hot(symbols, num_classes=vocab_size).float()
  loss_vals = -(one_hot * F.log_softmax(mean_log_probs, dim=-1)).sum(dim=-1)
  return (loss_vals * mask).sum().item(), mask.sum().item()


def retrain_step(models, retrain_optimizers, inputs, targets, current_lr):
  """Runs a single retraining step (forward + backward) over a full sequence with dropout."""
  losses = []
  for model, optimizer in zip(models, retrain_optimizers):
    optimizer.zero_grad()
    init_states = model.init_states(inputs.shape[0], inputs.device)
    # deterministic=False enables dropout (matches Flax deterministic=False path)
    with _lstm_autocast(inputs.device):
      logits, _ = model(inputs, init_states, return_sequence=True, deterministic=False)
    loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
        reduction='mean',
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
    # Match the JAX behavior: Adam with lr=1.0 and an explicit scaling by current_lr each step.
    # Setting param_groups[*]['lr'] = current_lr produces the same applied update for Adam.
    for g in optimizer.param_groups:
      g['lr'] = current_lr
    optimizer.step()
    losses.append(loss.item())
  return float(np.mean(losses))


def process(compress, length, vocab_size, coder, data):
  """Main processing loop for compression and decompression.

  Streams characters through the LSTM ensemble, periodically retrains on recent
  history, and drives the arithmetic coder one symbol at a time.

  When ``model_type == "transformer_xl"``, dispatches to the NNCP-style
  streaming loop in ``_process_transformer_xl`` instead. When ``model_type ==
  "hybrid"``, dispatches to the LSTM + Transformer-XL ensemble loop in
  ``_process_hybrid``.
  """
  _mt = globals().get('model_type', 'lstm')
  if _mt == 'transformer_xl':
    return _process_transformer_xl(compress, length, vocab_size, coder, data)
  if _mt == 'hybrid':
    return _process_hybrid(compress, length, vocab_size, coder, data)
  start = time.time()
  last_print_time = start
  reset_seed()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Optional TensorBoard logging (controlled by global params `tensorboard`,
  # `tensorboard_run_name`, and `tensorboard_logdir`).
  tb_writer = None
  if globals().get('tensorboard', False):
    from torch.utils.tensorboard import SummaryWriter
    _tb_logdir = globals().get('tensorboard_logdir', 'data/tensorboard')
    _tb_run = globals().get('tensorboard_run_name', 'torch_compress')
    tb_writer = SummaryWriter(log_dir=os.path.join(_tb_logdir, _tb_run))
    tb_writer.add_text('config', (
        f"backend=torch device={device} "
        f"batch_size={batch_size} seq_length={seq_length} "
        f"rnn_units={rnn_units} num_layers={num_layers} "
        f"embedding_size={embedding_size} ensemble_size={ensemble_size}"
    ))
    print(f"[tensorboard] writing to {tb_writer.log_dir}")

  lr_schedule = parse_schedule(learning_rate_schedule)
  retrain_p_schedule = parse_schedule(retrain_period_schedule)
  retrain_l_schedule = parse_schedule(retrain_lr_schedule)

  _ac_dtype, _ac_enabled = _resolve_precision()
  _ac_label = {torch.float16: 'fp16', torch.bfloat16: 'bf16',
               torch.float32: 'fp32'}[_ac_dtype]
  print(f"precision={_ac_label}")
  print(f"batch_size={batch_size}, seq_length={seq_length}, rnn_units={rnn_units}, num_layers={num_layers}, "
        f"embedding_size={embedding_size}, ensemble_size={ensemble_size}, learning_rate_schedule={learning_rate_schedule}, "
        f"adam_b1={adam_b1}, adam_b2={adam_b2}, adam_eps={adam_eps}, "
        f"retrain_period_schedule={retrain_period_schedule}, "
        f"retrain_block_len={retrain_block_len}, retrain_seq_length={retrain_seq_length}, "
        f"retrain_batch_size={retrain_batch_size}, retrain_lr_schedule={retrain_lr_schedule}, "
        f"retrain_dropout={retrain_dropout}, total_parts={total_parts}, current_part={current_part}")

  # Build ensemble
  models = []
  optimizers = []
  retrain_optimizers = []
  initial_lr = lr_schedule[0][1] if lr_schedule else 5e-4
  for _ in range(ensemble_size):
    m = build_model(globals().get('model_type', 'lstm'),
                    vocab_size=vocab_size, embedding_size=embedding_size,
                    rnn_units=rnn_units, num_layers=num_layers,
                    dropout_rate=retrain_dropout).to(device)
    models.append(m)
    optimizers.append(torch.optim.Adam(
        m.parameters(), lr=initial_lr, betas=(adam_b1, adam_b2), eps=adam_eps))
    retrain_optimizers.append(torch.optim.Adam(
        m.parameters(), lr=1.0, betas=(adam_b1, adam_b2), eps=adam_eps))

  # Restore model weights if a checkpoint exists
  ckpt_dir = os.path.abspath('data/ckpt')
  if checkpoint:
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
      ckpt = torch.load(ckpt_path, map_location=device)
      for i, m in enumerate(models):
        m.load_state_dict(ckpt['models'][i])
      print("Restored model weights from checkpoint.")
    else:
      print("No checkpoint found. Starting from scratch.")

  total_params = sum(p.numel() for p in models[0].parameters()) * ensemble_size
  print("\\n" + "=" * 80)
  print(f"Model Architecture (Ensemble Size: {ensemble_size})")
  print("=" * 80)
  print(models[0])
  print(f"\\nTotal Ensemble Parameters: {total_params:,}")
  print("=" * 80 + "\\n")

  split = math.ceil(length / batch_size)
  part_len = math.ceil(split / total_parts)
  start_pos = (current_part - 1) * part_len
  end_pos = min(start_pos + part_len, split)
  print(f"Processing part {current_part} of {total_parts}. Step range: {start_pos} to {end_pos - 1} (Total split: {split})")

  # Uniform prior used during the very first symbol of each stream.
  freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)

  pos = 0

  def fresh_states():
    return [m.init_states(batch_size, device) for m in models]

  model_state_loaded = False
  if start_pos > 0 and checkpoint:
    model_state_path = os.path.join(ckpt_dir, 'model_state.pt')
    if os.path.exists(model_state_path):
      print("Attempting to load model states from checkpoint...")
      try:
        loaded = torch.load(model_state_path, map_location=device)
        seq_input = loaded['seq_input'].to(device)
        if seq_input.shape != (batch_size, seq_length):
          raise ValueError(
              f"Loaded seq_input shape {tuple(seq_input.shape)} mismatch with batch/seq params.")
        states_queue = [
            [[(h.to(device), c.to(device)) for (h, c) in layer_states]
             for layer_states in step_states]
            for step_states in loaded['states_queue']
        ]
        if 'opt_states' in loaded:
          for opt, sd in zip(optimizers, loaded['opt_states']):
            opt.load_state_dict(sd)
          print("Restored main optimizer state.")
        if 'retrain_opt_states' in loaded:
          for opt, sd in zip(retrain_optimizers, loaded['retrain_opt_states']):
            opt.load_state_dict(sd)
          print("Restored retraining optimizer state.")
        print("Successfully loaded model states. Skipping warmup.")
        model_state_loaded = True
        pos = start_pos
      except Exception as e:
        print(f"Failed to load model states: {e}. Falling back to warmup.")

  if start_pos > 0 and not model_state_loaded:
    warmup_len = 500
    run_start_pos = max(0, start_pos - warmup_len)

    w_symbols = []
    for i in range(batch_size):
      start_idx_window = run_start_pos - seq_length + 1
      batch_syms = []
      for k in range(seq_length):
        idx_k = start_idx_window + k
        s_idx = idx_k + i * split
        if s_idx < 0:
          val = data[i * split] if i * split < len(data) else 0
        else:
          val = data[s_idx] if s_idx < len(data) else 0
        batch_syms.append(val)
      w_symbols.append(batch_syms)
    seq_input = torch.tensor(w_symbols, dtype=torch.long, device=device)

    states_queue = [fresh_states() for _ in range(seq_length)]

    print(f"Warming up states from {run_start_pos} to {start_pos}...")
    for w_pos in range(run_start_pos, start_pos):
      state_in = states_queue.pop(0)
      with torch.no_grad():
        new_state_per_model = []
        for model, states in zip(models, state_in):
          with _lstm_autocast(seq_input.device):
            _, ns = model(seq_input, states, return_sequence=False, deterministic=True)
          new_state_per_model.append(ns)
      states_queue.append(detach_states(new_state_per_model))

      next_in_symbols = []
      for i in range(batch_size):
        idx = w_pos + 1 + i * split
        val = data[idx] if idx < len(data) else 0
        next_in_symbols.append(val)
      next_in_t = torch.tensor(next_in_symbols, dtype=torch.long, device=device).unsqueeze(1)
      seq_input = torch.cat([seq_input[:, 1:], next_in_t], dim=1)

    pos = start_pos

  elif not model_state_loaded:
    initial_symbols = []
    for i in range(batch_size):
      initial_symbols.append(get_symbol(i * split, length, freq, coder, compress, data))
    seq_input = torch.tensor(initial_symbols, dtype=torch.long, device=device).unsqueeze(1).repeat(1, seq_length)
    states_queue = [fresh_states() for _ in range(seq_length)]
    pos = 0

  cross_entropy = 0.0
  denom = 0.0
  template = '{:0.2f}%\\tcross entropy: {:0.2f}\\ttime: {:0.2f}\\tlr: {:0.8f}\\tstep: {}'

  last_retrain_pos = 0
  current_lr = initial_lr

  while pos < end_pos:
    # Update online learning rate
    current_lr = get_scheduled_value(lr_schedule, pos)
    for opt in optimizers:
      for g in opt.param_groups:
        g['lr'] = current_lr

    current_retrain_period = get_scheduled_value(retrain_p_schedule, pos)
    current_retrain_lr = get_scheduled_value(retrain_l_schedule, pos)

    if current_retrain_period > 0 and (pos - last_retrain_pos) >= current_retrain_period:
      retrain_start_time = time.time()

      r_start_step = max(0, pos - retrain_block_len)
      r_end_step = pos

      all_inputs = []
      all_targets = []

      r_step = r_start_step
      while r_step < r_end_step:
        for i in range(batch_size):
          base_idx = r_step + i * split
          start_idx = base_idx
          end_idx = start_idx + retrain_seq_length + 1
          current_stream_limit = i * split + pos + 1

          stream_segment = data[start_idx: min(end_idx, current_stream_limit)]
          if len(stream_segment) < retrain_seq_length + 1:
            stream_segment = list(stream_segment) + [0] * (retrain_seq_length + 1 - len(stream_segment))

          all_inputs.append(stream_segment[:-1])
          all_targets.append(stream_segment[1:])

        r_step += retrain_seq_length

      all_inputs_t = torch.tensor(all_inputs, dtype=torch.long, device=device)
      all_targets_t = torch.tensor(all_targets, dtype=torch.long, device=device)

      total_examples = all_inputs_t.shape[0]
      remainder = total_examples % retrain_batch_size
      if remainder != 0:
        all_inputs_t = all_inputs_t[:-remainder]
        all_targets_t = all_targets_t[:-remainder]
        total_examples -= remainder

      print(f"Starting retraining at step {pos}... (period={current_retrain_period:.1f}, "
            f"lr={current_retrain_lr:.8f}, examples={total_examples})")

      retrain_loss_sum = 0.0
      retrain_batches = 0
      if total_examples > 0:
        for i in range(0, total_examples, retrain_batch_size):
          batch_inputs = all_inputs_t[i: i + retrain_batch_size]
          batch_targets = all_targets_t[i: i + retrain_batch_size]
          rl = retrain_step(models, retrain_optimizers, batch_inputs, batch_targets, current_retrain_lr)
          retrain_loss_sum += rl
          retrain_batches += 1

      last_retrain_pos = pos
      retrain_duration = time.time() - retrain_start_time
      print(f"Retraining finished. Duration: {retrain_duration:.2f}s")
      if tb_writer is not None and retrain_batches > 0:
        tb_writer.add_scalar('retrain/loss', retrain_loss_sum / retrain_batches, pos)
        tb_writer.add_scalar('retrain/lr', current_retrain_lr, pos)
        tb_writer.add_scalar('retrain/duration_sec', retrain_duration, pos)

    state_in = states_queue.pop(0)

    # Forward pass (autograd enabled, dropout disabled to match deterministic=True)
    probs_np, logits_list, new_states = forward_ensemble(models, seq_input, state_in)

    # Drive the arithmetic coder symbol-by-symbol
    current_symbols = []
    current_mask = []
    for i in range(batch_size):
      p_i = probs_np[i]
      freq_i = np.cumsum(p_i * 10000000 + 1)
      index = pos + 1 + i * split
      symbol = get_symbol(index, length, freq_i, coder, compress, data)
      current_symbols.append(symbol)
      current_mask.append(1.0 if index < length else 0.0)

    symbols_t = torch.tensor(current_symbols, dtype=torch.long, device=device)
    mask_t = torch.tensor(current_mask, dtype=torch.float32, device=device)

    loss_val, loss_denom = backward_and_step(
        logits_list, optimizers, models, symbols_t, mask_t, vocab_size)

    cross_entropy += loss_val
    denom += loss_denom

    states_queue.append(detach_states(new_states))

    seq_input = torch.cat([seq_input[:, 1:], symbols_t.unsqueeze(1)], dim=1)
    pos += 1

    if time.time() - last_print_time >= 20:
      last_print_time = time.time()
      time_diff = last_print_time - start
      current_bpc = (cross_entropy / denom) / np.log(2)
      print(template.format(pos / split * 100, current_bpc, time_diff, current_lr, pos))
      if tb_writer is not None:
        tb_writer.add_scalar('train/bpc', current_bpc, pos)
        tb_writer.add_scalar('train/cross_entropy_total', cross_entropy, pos)
        tb_writer.add_scalar('train/lr', current_lr, pos)
        tb_writer.add_scalar('train/elapsed_sec', time_diff, pos)
        tb_writer.add_scalar('train/steps_per_sec', pos / time_diff if time_diff > 0 else 0.0, pos)

  if checkpoint:
    print("Saving checkpoint...")
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

    torch.save({
        'models': [m.state_dict() for m in models],
        'opt_states': [o.state_dict() for o in optimizers],
        'retrain_opt_states': [o.state_dict() for o in retrain_optimizers],
    }, os.path.join(ckpt_dir, 'checkpoint.pt'))
    print("Checkpoint saved.")

    ac_state = {
        'coder': coder.get_state(),
        'bitstream': coder.output.get_state() if compress else coder.input.get_state(),
    }
    if not compress:
      ac_state['file_pos'] = coder.input.input.tell()
    with open(os.path.join(ckpt_dir, 'ac_state.pkl'), 'wb') as f:
      pickle.dump(ac_state, f)
    print("AC state saved.")

    seq_input_cpu = seq_input.detach().cpu()
    states_queue_cpu = [
        [[(h.detach().cpu(), c.detach().cpu()) for (h, c) in layer_states]
         for layer_states in step_states]
        for step_states in states_queue
    ]
    torch.save({
        'seq_input': seq_input_cpu,
        'states_queue': states_queue_cpu,
        'opt_states': [o.state_dict() for o in optimizers],
        'retrain_opt_states': [o.state_dict() for o in retrain_optimizers],
    }, os.path.join(ckpt_dir, 'model_state.pt'))
    print("Model state (including optimizers) saved.")

  if tb_writer is not None:
    final_time = time.time() - start
    if denom > 0:
      tb_writer.add_scalar('train/bpc', (cross_entropy / denom) / np.log(2), pos)
    tb_writer.add_scalar('train/elapsed_sec', final_time, pos)
    tb_writer.add_text('summary', (
        f"final_pos={pos} elapsed_sec={final_time:.2f} "
        f"final_bpc={(cross_entropy / denom) / np.log(2) if denom > 0 else float('nan'):.4f}"
    ))
    tb_writer.flush()
    tb_writer.close()


def _process_transformer_xl(compress, length, vocab_size, coder, data):
  """Transformer-XL streaming compress/decompress, modeled on NNCP v2's train().

  Differences from the LSTM path:
  - Mems carry forward naturally between steps; there is no state queue / BPTT.
  - Each step feeds (ext_tgt_len + 1) tokens and predicts the last token.
    (Phase 2A runs with tgt_len=1 only — multi-target packing is Phase 2B.)
  - Online dropout is off (deterministic=True), matching the LSTM path.

  Online retraining honors ``retrain_period_schedule`` and runs in NNCP's
  retrain shape (``retrain_tgt_len``, ``retrain_mem_len``) via
  ``_retrain_transformer_xl``; ``model.reset_length`` toggles around the call.

  Mixed-precision: when ``use_bf16`` is True, forward + backward run under
  ``torch.autocast`` with bfloat16 activations while the model parameters and
  optimizer state stay in fp32 (standard mixed-precision pattern). Logits are
  cast back to fp32 before the AC encoder consumes them so the arithmetic
  coding maths is identical to the fp32 path. On GH200 / H100 this is ~10-15x
  faster than fp32 for the large NNCP-base config; on CPU it has no
  meaningful effect.

  Limitations (later work):
  - Multi-part / checkpointing not yet supported (asserts current_part == 1).
  """
  assert current_part == 1 and total_parts == 1, (
      "transformer_xl backend currently supports only single-part runs."
  )
  if checkpoint:
    print("[transformer_xl] Warning: checkpoint=True is not yet supported; ignoring.")

  # bf16 mixed precision is round-trip safe given the
  # allow_bf16_reduced_precision_reduction = False flag set in IMPORTS_SRC.
  # Without that flag, bf16 matmul kernels accumulated reductions in bf16
  # with non-deterministic thread ordering, breaking encode/decode agreement
  # on probabilities during retrain (see commit 3f6a620 for the diagnostic).
  ac_dtype, use_bf16_flag = _resolve_precision()  # var name kept; "enabled" semantics

  start = time.time()
  last_print_time = start
  reset_seed()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  tb_writer = None
  if globals().get('tensorboard', False):
    from torch.utils.tensorboard import SummaryWriter
    _tb_logdir = globals().get('tensorboard_logdir', 'data/tensorboard')
    _tb_run = globals().get('tensorboard_run_name', 'torch_compress')
    tb_writer = SummaryWriter(log_dir=os.path.join(_tb_logdir, _tb_run))
    tb_writer.add_text('config', (
        f"backend=torch model=transformer_xl device={device} "
        f"batch_size={batch_size} ext_tgt_len={ext_tgt_len} mem_len={mem_len} "
        f"n_layer={n_layer} d_model={d_model} n_head={n_head} d_head={d_head} "
        f"d_inner={d_inner} ensemble_size={ensemble_size}"
    ))
    print(f"[tensorboard] writing to {tb_writer.log_dir}")

  # Transformer-XL uses its own LR schedules (NNCP-base defaults). The LSTM-
  # tuned `learning_rate_schedule` / `retrain_lr_schedule` stay reserved for
  # the LSTM path -- transformers want lower LRs and a different decay shape.
  lr_schedule_str = globals().get('learning_rate_schedule_xl', learning_rate_schedule)
  lr_schedule = parse_schedule(lr_schedule_str)

  print(f"[transformer_xl] batch_size={batch_size}, ext_tgt_len={ext_tgt_len}, "
        f"mem_len={mem_len}, n_layer={n_layer}, d_model={d_model}, n_head={n_head}, "
        f"d_head={d_head}, d_inner={d_inner}, ensemble_size={ensemble_size}, "
        f"learning_rate_schedule_xl={lr_schedule_str}, "
        f"adam_b1={adam_b1}, adam_b2={adam_b2}, adam_eps={adam_eps}")

  models = []
  optimizers = []
  retrain_optimizers = []
  initial_lr = lr_schedule[0][1] if lr_schedule else 5e-4
  # NNCP-aligned Adam epsilon (1e-9 vs torch_compress's tuned 1e-12 for LSTM).
  _adam_eps_xl = float(globals().get('adam_eps_xl', 1e-9))
  for _ in range(ensemble_size):
    m = build_model('transformer_xl',
                    vocab_size=vocab_size, embedding_size=embedding_size,
                    rnn_units=rnn_units, num_layers=num_layers,
                    dropout_rate=0.0).to(device)
    models.append(m)
    optimizers.append(torch.optim.Adam(
        m.parameters(), lr=initial_lr, betas=(adam_b1, adam_b2), eps=_adam_eps_xl))
    # Separate optimizer state for retraining, mirroring NNCP's saved-state
    # toggle and torch_compress's two-optimizer pattern. lr is overwritten
    # per-step inside the retrain function.
    retrain_optimizers.append(torch.optim.Adam(
        m.parameters(), lr=1.0, betas=(adam_b1, adam_b2), eps=_adam_eps_xl))

  total_params = sum(p.numel() for p in models[0].parameters()) * ensemble_size
  print("\\n" + "=" * 80)
  print(f"Transformer-XL Architecture (Ensemble Size: {ensemble_size})")
  print("=" * 80)
  print(models[0])
  print(f"\\nTotal Ensemble Parameters: {total_params:,}")
  print("=" * 80 + "\\n")

  split = math.ceil(length / batch_size)
  end_pos = split
  print(f"Processing single part. Stream length per batch: {split}")

  # Uniform prior used to AC-code the very first symbol of each parallel stream.
  freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)

  qlen = ext_tgt_len + 1  # tgt_len = 1 for streaming inference

  initial_symbols = []
  for i in range(batch_size):
    initial_symbols.append(get_symbol(i * split, length, freq, coder, compress, data))
  # Bootstrap input window: the just-coded symbol replicated qlen times.
  # Real context accumulates as the stream advances.
  seq_input = (
      torch.tensor(initial_symbols, dtype=torch.long, device=device)
      .unsqueeze(1)
      .repeat(1, qlen)
  )
  mems_per_model = _xl_init_mems(models, batch_size, device, ac_dtype)

  retrain_p_schedule = parse_schedule(retrain_period_schedule)
  retrain_l_schedule_str = globals().get('retrain_lr_schedule_xl', retrain_lr_schedule)
  retrain_l_schedule = parse_schedule(retrain_l_schedule_str)

  cross_entropy = 0.0
  denom = 0.0
  template = '{:0.2f}%\\tcross entropy: {:0.2f}\\ttime: {:0.2f}\\tlr: {:0.8f}\\tstep: {}'

  pos = 0
  current_lr = initial_lr
  last_retrain_pos = 0

  while pos < end_pos:
    current_lr = get_scheduled_value(lr_schedule, pos)
    for opt in optimizers:
      for g in opt.param_groups:
        g['lr'] = current_lr

    # NNCP-style retrain: reshape model to retrain_(tgt_len, mem_len), run a
    # streaming pass over the trailing retrain_block_len chars with mems
    # carrying within the pass, then restore the streaming shape.
    current_retrain_period = get_scheduled_value(retrain_p_schedule, pos)
    current_retrain_lr = get_scheduled_value(retrain_l_schedule, pos)
    if current_retrain_period > 0 and (pos - last_retrain_pos) >= current_retrain_period:
      retrain_start_time = time.time()
      retrain_loss = _retrain_transformer_xl(
          models=models,
          retrain_optimizers=retrain_optimizers,
          current_lr=current_retrain_lr,
          file_data=data,
          file_pos=pos + 1,
          retrain_block_len=retrain_block_len,
          retrain_tgt_len=retrain_tgt_len,
          retrain_mem_len=retrain_mem_len,
          retrain_batch_size=retrain_batch_size,
          stream_mem_len=mem_len,
          vocab_size=vocab_size,
          device=device,
      )
      last_retrain_pos = pos
      retrain_duration = time.time() - retrain_start_time
      print(f"[transformer_xl] retrain done at step {pos}: "
            f"loss={retrain_loss:.4f}, duration={retrain_duration:.2f}s")
      if tb_writer is not None:
        tb_writer.add_scalar('retrain/loss', retrain_loss, pos)
        tb_writer.add_scalar('retrain/lr', current_retrain_lr, pos)
        tb_writer.add_scalar('retrain/duration_sec', retrain_duration, pos)
      # Mems were attached to the pre-retrain stream-shape model; the retrain
      # mutated weights and toggled reset_length back to streaming, but the
      # mems still reflect activations from before the weight update. Reset
      # them to empty so the next forward pass re-builds context with the
      # updated weights.
      mems_per_model = _xl_init_mems(models, batch_size, device, ac_dtype)

    # Forward — autograd ON, dropout off (matches LSTM forward_ensemble path).
    log_probs_list = []
    logits_list = []
    new_mems_list = []
    for model, mems in zip(models, mems_per_model):
      with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=use_bf16_flag):
        logits, new_mems = model(seq_input, mems, return_sequence=False, deterministic=True)
      # AC consumes fp32 probabilities; cast logits up so the loss + AC math
      # below is fp32 even when the forward ran in bf16.
      logits = logits.float()
      logits_list.append(logits)
      log_probs_list.append(F.log_softmax(logits, dim=-1))
      new_mems_list.append(new_mems)
    mean_log_probs = torch.stack(log_probs_list, dim=0).mean(dim=0)
    probs = F.softmax(mean_log_probs, dim=-1).detach().cpu().numpy()

    # Drive the arithmetic coder: one symbol per parallel stream.
    current_symbols = []
    current_mask = []
    for i in range(batch_size):
      freq_i = np.cumsum(probs[i] * 10000000 + 1)
      index = pos + 1 + i * split
      symbol = get_symbol(index, length, freq_i, coder, compress, data)
      current_symbols.append(symbol)
      current_mask.append(1.0 if index < length else 0.0)

    symbols_t = torch.tensor(current_symbols, dtype=torch.long, device=device)
    mask_t = torch.tensor(current_mask, dtype=torch.float32, device=device)

    # Reuse the LSTM-side helper — same per-step loss/clip/step shape.
    # NNCP-aligned: pass clip_xl (default 0.25) instead of the LSTM's 4.0.
    loss_val, loss_denom = backward_and_step(
        logits_list, optimizers, models, symbols_t, mask_t, vocab_size,
        clip=float(globals().get('clip_xl', 0.25)))
    cross_entropy += loss_val
    denom += loss_denom

    # Carry mems forward (already detached inside MemTransformerLM._update_mems).
    mems_per_model = new_mems_list
    # Slide window: drop oldest, append the symbol just coded.
    seq_input = torch.cat([seq_input[:, 1:], symbols_t.unsqueeze(1)], dim=1)
    pos += 1

    if time.time() - last_print_time >= 20:
      last_print_time = time.time()
      time_diff = last_print_time - start
      current_bpc = (cross_entropy / denom) / np.log(2)
      print(template.format(pos / split * 100, current_bpc, time_diff, current_lr, pos))
      if tb_writer is not None:
        tb_writer.add_scalar('train/bpc', current_bpc, pos)
        tb_writer.add_scalar('train/cross_entropy_total', cross_entropy, pos)
        tb_writer.add_scalar('train/lr', current_lr, pos)
        tb_writer.add_scalar('train/elapsed_sec', time_diff, pos)
        tb_writer.add_scalar('train/steps_per_sec', pos / time_diff if time_diff > 0 else 0.0, pos)

  if tb_writer is not None:
    final_time = time.time() - start
    if denom > 0:
      tb_writer.add_scalar('train/bpc', (cross_entropy / denom) / np.log(2), pos)
    tb_writer.add_scalar('train/elapsed_sec', final_time, pos)
    tb_writer.add_text('summary', (
        f"final_pos={pos} elapsed_sec={final_time:.2f} "
        f"final_bpc={(cross_entropy / denom) / np.log(2) if denom > 0 else float('nan'):.4f}"
    ))
    tb_writer.flush()
    tb_writer.close()


def _xl_init_mems(models, batch_size, device, ac_dtype):
  """Build empty mems for each ensemble member, casting to ``ac_dtype``.

  ``MemTransformerLM.init_mems`` returns tensors of the model's parameter
  dtype (fp32). Under bf16 mixed-precision the first forward pass produces
  bf16 hidden states; ``_update_mems`` then concatenates the (fp32) initial
  empty mems with the (bf16) hids, which would error on dtype mismatch in
  ``torch.cat``. Pre-casting here keeps the mems in the right dtype from the
  start, and is a no-op for fp32.
  """
  out = []
  for m in models:
    mems = m.init_states(batch_size, device)
    if mems is None:
      out.append(None)
      continue
    if ac_dtype is not None and ac_dtype != torch.float32:
      mems = [t.to(ac_dtype) for t in mems]
    out.append(mems)
  return out


def _retrain_transformer_xl(*, models, retrain_optimizers, current_lr, file_data,
                            file_pos, retrain_block_len, retrain_tgt_len,
                            retrain_mem_len, retrain_batch_size, stream_mem_len,
                            vocab_size, device):
  """NNCP-style retraining pass on the trailing ``retrain_block_len`` chars.

  Mirrors NNCP v2's ``retrain()`` from ``nncp.py``:

  - ``model.reset_length(retrain_tgt_len, 0, retrain_mem_len)`` reshapes the
    model for retrain (typically tgt_len=64, mem_len=128 for NNCP-base).
  - The retrain window is split into ``retrain_batch_size`` parallel streams,
    each ``block_stride = block_len // retrain_batch_size`` chars long.
  - Steps process ``retrain_tgt_len`` tokens at a time; mems carry forward
    *within* the retrain pass (this is the part that distinguishes NNCP-style
    retrain from the LSTM-style fresh-state-per-batch retrain).
  - Dropout is on (``deterministic=False`` -> ``model.train()`` inside the
    adapter); the model's dropout rate was set at construction via the
    ``dropout`` notebook hparam.
  - On the very first step, the input row is dummied (matches NNCP exactly --
    avoids the off-by-one negative index at stream_pos=0).
  - At exit, ``model.reset_length(1, 0, stream_mem_len)`` restores the
    streaming-inference shape so the caller's ``mems_per_model`` rebuild
    (which the caller does immediately after) produces correctly-sized mems.

  Returns the mean per-step cross-entropy loss across the ensemble (in nats).
  """
  block_len = min(file_pos, retrain_block_len)
  if block_len < retrain_batch_size * retrain_tgt_len:
    return 0.0  # not enough trailing data for a single retrain step
  block_start = file_pos - block_len
  block_stride = block_len // retrain_batch_size
  if block_stride < retrain_tgt_len:
    return 0.0

  # Switch every ensemble member to retrain shape before the first forward.
  for model in models:
    model.reset_length(retrain_tgt_len, 0, retrain_mem_len)

  ac_dtype, use_bf16_flag = _resolve_precision()  # var name kept; "enabled" semantics

  ensemble_losses = []
  try:
    for model, optimizer in zip(models, retrain_optimizers):
      mems = model.init_states(retrain_batch_size, device)
      if mems is not None and ac_dtype != torch.float32:
        mems = [t.to(ac_dtype) for t in mems]
      stream_pos = 0
      step_losses = []
      while (stream_pos + retrain_tgt_len) <= block_stride:
        data0 = []
        target0 = []
        for j in range(retrain_tgt_len):
          target_pos = block_start + stream_pos + j
          target_row = file_data[target_pos: target_pos + block_stride * retrain_batch_size: block_stride]
          target0.append(target_row)
          if stream_pos == 0:
            data_row = [0] * retrain_batch_size  # dummy first step (matches NNCP)
          else:
            input_pos = target_pos - 1
            data_row = file_data[input_pos: input_pos + block_stride * retrain_batch_size: block_stride]
          data0.append(data_row)

        # Build (retrain_tgt_len, retrain_batch_size) tensors then transpose
        # to (batch, tgt_len) to match the adapter's batch-first interface.
        data_t = torch.tensor(data0, dtype=torch.long, device=device).t().contiguous()
        target_t = torch.tensor(target0, dtype=torch.long, device=device).t().contiguous()

        optimizer.zero_grad()
        # return_sequence=True -> logits at every position; deterministic=False
        # -> dropout on (model.train() set by the adapter).
        with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=use_bf16_flag):
          logits, mems = model(data_t, mems, return_sequence=True, deterministic=False)
        # Cast to fp32 so cross_entropy + backward run in fp32 (matches the
        # streaming loop's pattern).
        logits = logits.float()
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_t.reshape(-1),
            reduction='mean',
        )
        loss.backward()
        # NNCP-aligned clip (0.25) for the transformer retrain pass.
        torch.nn.utils.clip_grad_norm_(model.parameters(),
            float(globals().get('clip_xl', 0.25)))
        for g in optimizer.param_groups:
          g['lr'] = current_lr
        optimizer.step()
        step_losses.append(loss.item())
        stream_pos += retrain_tgt_len

      ensemble_losses.append(float(np.mean(step_losses)) if step_losses else 0.0)
  finally:
    # Always restore the streaming shape, even on exception.
    for model in models:
      model.reset_length(1, 0, stream_mem_len)

  return float(np.mean(ensemble_losses)) if ensemble_losses else 0.0


def _process_hybrid(compress, length, vocab_size, coder, data):
  """LSTM + Transformer-XL hybrid streaming loop with geometric-mean ensembling.

  At every step both submodels forward against the same just-coded symbol
  context. AC sees ``softmax(0.5 * (log_softmax(lstm_logits) +
  log_softmax(xl_logits)))`` -- the geometric mean of the two distributions.
  Each submodel backprops independently against the actual symbol with its
  own optimizer (and its own LR schedule).

  Per-submodel state management mirrors each model's solo loop:
    - LSTM: states_queue of seq_length snapshots, BPTT replay through the
      window, push detached state after the forward.
    - Transformer-XL: mems list, carries forward via _update_mems inside the
      model. Pre-cast to bf16 if use_bf16; reset to empty after retrain.

  Limitations:
  - ensemble_size > 1 is rejected here -- each "ensemble member" is itself a
    hybrid pair, and stacking multiple hybrids would double-count parameters
    in unhelpful ways. Use a separate run with model_type="lstm" or
    "transformer_xl" if you want to homogeneous-ensemble.
  - Multi-part / checkpointing not supported (asserts current_part == 1).
  """
  assert current_part == 1 and total_parts == 1, (
      "hybrid backend currently supports only single-part runs."
  )
  assert ensemble_size == 1, (
      "hybrid backend currently supports only ensemble_size=1; the hybrid "
      "itself is already a 2-model ensemble."
  )
  if checkpoint:
    print("[hybrid] Warning: checkpoint=True is not yet supported; ignoring.")

  # bf16 mixed precision: applies only to the Transformer-XL submodel's
  # forward (LSTM cell loop runs in fp32 regardless -- bf16 isn't wired into
  # LSTMModel and the BPTT compounding makes it risky there).
  ac_dtype, use_bf16_flag = _resolve_precision()  # var name kept; "enabled" semantics

  start = time.time()
  last_print_time = start
  reset_seed()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  tb_writer = None
  if globals().get('tensorboard', False):
    from torch.utils.tensorboard import SummaryWriter
    _tb_logdir = globals().get('tensorboard_logdir', 'data/tensorboard')
    _tb_run = globals().get('tensorboard_run_name', 'torch_compress')
    tb_writer = SummaryWriter(log_dir=os.path.join(_tb_logdir, _tb_run))
    tb_writer.add_text('config', (
        f"backend=torch model=hybrid device={device} "
        f"batch_size={batch_size} seq_length={seq_length} "
        f"rnn_units={rnn_units} num_layers={num_layers} "
        f"embedding_size={embedding_size} ext_tgt_len={ext_tgt_len} "
        f"mem_len={mem_len} n_layer={n_layer} d_model={d_model}"
    ))

  # Two LR schedules: LSTM uses learning_rate_schedule (its tuned default);
  # Transformer-XL uses learning_rate_schedule_xl (NNCP-base default).
  lstm_lr_schedule = parse_schedule(learning_rate_schedule)
  xl_lr_schedule_str = globals().get('learning_rate_schedule_xl', learning_rate_schedule)
  xl_lr_schedule = parse_schedule(xl_lr_schedule_str)
  retrain_p_schedule = parse_schedule(retrain_period_schedule)
  retrain_l_schedule = parse_schedule(retrain_lr_schedule)
  retrain_l_schedule_xl_str = globals().get('retrain_lr_schedule_xl', retrain_lr_schedule)
  retrain_l_schedule_xl = parse_schedule(retrain_l_schedule_xl_str)

  initial_lstm_lr = lstm_lr_schedule[0][1] if lstm_lr_schedule else 5e-4
  initial_xl_lr = xl_lr_schedule[0][1] if xl_lr_schedule else 7.9e-5

  # Build hybrid model + per-submodel optimizers.
  model = build_model('hybrid', vocab_size=vocab_size, embedding_size=embedding_size,
                      rnn_units=rnn_units, num_layers=num_layers,
                      dropout_rate=retrain_dropout).to(device)

  # LSTM half keeps torch_compress's tuned adam_eps; Transformer-XL half
  # uses NNCP-aligned adam_eps_xl. Mixer keeps adam_eps (the LSTM-side value
  # is fine for a tiny gating MLP).
  _adam_eps_xl = float(globals().get('adam_eps_xl', 1e-9))
  lstm_opt = torch.optim.Adam(
      model.lstm.parameters(), lr=initial_lstm_lr, betas=(adam_b1, adam_b2), eps=adam_eps)
  xl_opt = torch.optim.Adam(
      model.transformer_xl.parameters(), lr=initial_xl_lr, betas=(adam_b1, adam_b2), eps=_adam_eps_xl)
  # Separate retrain optimizer state per submodel (lr=1.0 placeholder; per-step set).
  lstm_retrain_opt = torch.optim.Adam(
      model.lstm.parameters(), lr=1.0, betas=(adam_b1, adam_b2), eps=adam_eps)
  xl_retrain_opt = torch.optim.Adam(
      model.transformer_xl.parameters(), lr=1.0, betas=(adam_b1, adam_b2), eps=_adam_eps_xl)

  # Optional learned mixer optimizer. The mixer is tiny (~300 params) so it
  # can take a much higher LR than the submodels; defaults to 0.01 unless
  # overridden via mixer_lr global.
  mixer_opt = None
  if model.mixer is not None:
    mixer_lr = globals().get('mixer_lr', 0.01)
    mixer_opt = torch.optim.Adam(
        model.mixer.parameters(), lr=mixer_lr, betas=(adam_b1, adam_b2), eps=adam_eps)

  total_params = sum(p.numel() for p in model.parameters())
  lstm_params = sum(p.numel() for p in model.lstm.parameters())
  xl_params = sum(p.numel() for p in model.transformer_xl.parameters())
  mixer_params = sum(p.numel() for p in model.mixer.parameters()) if model.mixer is not None else 0
  print("\\n" + "=" * 80)
  print(f"Hybrid LSTM + Transformer-XL" + (" + LearnedMixer" if model.mixer is not None else ""))
  print("=" * 80)
  print(f"LSTM submodel parameters:           {lstm_params:,}")
  print(f"Transformer-XL submodel parameters: {xl_params:,}")
  if model.mixer is not None:
    print(f"LearnedMixer parameters:            {mixer_params:,}")
  print(f"Total parameters:                   {total_params:,}")
  print("=" * 80 + "\\n")

  split = math.ceil(length / batch_size)
  end_pos = split

  # Uniform prior used to AC-code the very first symbol of each parallel stream.
  freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)

  qlen_xl = ext_tgt_len + 1  # Transformer streaming window
  # LSTM streaming window is `seq_length` (same param the LSTM solo path uses).

  initial_symbols = []
  for i in range(batch_size):
    initial_symbols.append(get_symbol(i * split, length, freq, coder, compress, data))

  # Two seq_input windows -- LSTM consumes `seq_length` tokens per step (with
  # BPTT), Transformer-XL consumes `qlen_xl = ext_tgt_len + 1`. They slide in
  # lockstep but can be different widths.
  base = torch.tensor(initial_symbols, dtype=torch.long, device=device).unsqueeze(1)
  lstm_seq_input = base.repeat(1, seq_length)
  xl_seq_input = base.repeat(1, qlen_xl)

  def fresh_lstm_states():
    return model.lstm.init_states(batch_size, device)

  lstm_states_queue = [fresh_lstm_states() for _ in range(seq_length)]

  xl_mems = model.transformer_xl.init_states(batch_size, device)
  if xl_mems is not None and ac_dtype != torch.float32:
    xl_mems = [t.to(ac_dtype) for t in xl_mems]

  cross_entropy = 0.0
  denom = 0.0
  template = '{:0.2f}%\\tcross entropy: {:0.2f}\\ttime: {:0.2f}\\tlstm_lr: {:0.6f}\\txl_lr: {:0.6f}\\tstep: {}'

  pos = 0
  current_lstm_lr = initial_lstm_lr
  current_xl_lr = initial_xl_lr
  last_retrain_pos = 0

  while pos < end_pos:
    # Per-step LR updates (each submodel uses its own schedule).
    current_lstm_lr = get_scheduled_value(lstm_lr_schedule, pos)
    current_xl_lr = get_scheduled_value(xl_lr_schedule, pos)
    for g in lstm_opt.param_groups: g['lr'] = current_lstm_lr
    for g in xl_opt.param_groups: g['lr'] = current_xl_lr

    # Periodic retrain (both submodels retrain on the same retrain window in
    # their respective shapes). Mems / states are reset after retrain since
    # the weights changed underneath them.
    current_retrain_period = get_scheduled_value(retrain_p_schedule, pos)
    current_lstm_retrain_lr = get_scheduled_value(retrain_l_schedule, pos)
    current_xl_retrain_lr = get_scheduled_value(retrain_l_schedule_xl, pos)
    if current_retrain_period > 0 and (pos - last_retrain_pos) >= current_retrain_period:
      retrain_start_time = time.time()
      _retrain_hybrid(
          model=model,
          lstm_retrain_opt=lstm_retrain_opt,
          xl_retrain_opt=xl_retrain_opt,
          lstm_lr=current_lstm_retrain_lr,
          xl_lr=current_xl_retrain_lr,
          file_data=data,
          file_pos=pos + 1,
          split=split,
          batch_size_streaming=batch_size,
          retrain_block_len=retrain_block_len,
          retrain_seq_length=retrain_seq_length,
          retrain_batch_size_lstm=retrain_batch_size,
          retrain_tgt_len=retrain_tgt_len,
          retrain_mem_len=retrain_mem_len,
          stream_mem_len=mem_len,
          vocab_size=vocab_size,
          device=device,
      )
      last_retrain_pos = pos
      retrain_duration = time.time() - retrain_start_time
      print(f"[hybrid] retrain done at step {pos}: duration={retrain_duration:.2f}s")
      if tb_writer is not None:
        tb_writer.add_scalar('retrain/duration_sec', retrain_duration, pos)
        tb_writer.add_scalar('retrain/lstm_lr', current_lstm_retrain_lr, pos)
        tb_writer.add_scalar('retrain/xl_lr', current_xl_retrain_lr, pos)
      # Reset both submodels' running state -- pre-retrain activations are
      # stale relative to the post-retrain weights.
      lstm_states_queue = [fresh_lstm_states() for _ in range(seq_length)]
      xl_mems = model.transformer_xl.init_states(batch_size, device)
      if xl_mems is not None and ac_dtype != torch.float32:
        xl_mems = [t.to(ac_dtype) for t in xl_mems]

    # ---- Forward both submodels ----
    # LSTM: pop the oldest state from queue, BPTT replay over seq_length
    lstm_state_in = lstm_states_queue.pop(0)
    lstm_logits, new_lstm_state = model.lstm(
        lstm_seq_input, lstm_state_in, return_sequence=False, deterministic=True)

    # Transformer-XL: forward with current mems under autocast
    with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=use_bf16_flag):
      xl_logits, new_xl_mems = model.transformer_xl(
          xl_seq_input, xl_mems, return_sequence=False, deterministic=True)
    xl_logits = xl_logits.float()

    # ---- Combine: equal-weight or learned-mixer geometric mean ----
    log_probs_lstm = F.log_softmax(lstm_logits, dim=-1)
    log_probs_xl = F.log_softmax(xl_logits, dim=-1)
    if model.mixer is not None:
      # Learned mixer: per-step softmax weights from confidence features.
      # Output is generally unnormalised; log_softmax it to get a valid
      # log-prob distribution for the AC.
      combined_unnorm = model.mixer([log_probs_lstm, log_probs_xl])
      combined_log_probs = F.log_softmax(combined_unnorm, dim=-1)
    else:
      # Equal-weight geometric mean (the original hybrid behaviour).
      combined_log_probs = 0.5 * (log_probs_lstm + log_probs_xl)
    probs = F.softmax(combined_log_probs, dim=-1).detach().cpu().numpy()

    # ---- Drive the AC: one symbol per parallel stream ----
    current_symbols = []
    current_mask = []
    for i in range(batch_size):
      freq_i = np.cumsum(probs[i] * 10000000 + 1)
      index = pos + 1 + i * split
      symbol = get_symbol(index, length, freq_i, coder, compress, data)
      current_symbols.append(symbol)
      current_mask.append(1.0 if index < length else 0.0)

    symbols_t = torch.tensor(current_symbols, dtype=torch.long, device=device)
    mask_t = torch.tensor(current_mask, dtype=torch.float32, device=device)

    # ---- Backward ----
    # Each submodel always gets gradient from its own solo loss (so each
    # learns to be a competent solo predictor regardless of the mixer).
    # If a learned mixer is in play, an additional ensemble-loss term is
    # added: cross-entropy of the mixer's combined distribution against the
    # actual symbol. Gradient from the mixer-loss flows through the mixer
    # AND back through the submodels, so they co-adapt to be good ensemble
    # components in addition to good solos.
    lstm_opt.zero_grad()
    xl_opt.zero_grad()
    if mixer_opt is not None:
      mixer_opt.zero_grad()

    lstm_loss_per = F.cross_entropy(lstm_logits, symbols_t, reduction='none')
    xl_loss_per = F.cross_entropy(xl_logits, symbols_t, reduction='none')
    total_loss = (lstm_loss_per * mask_t).sum() + (xl_loss_per * mask_t).sum()

    if model.mixer is not None:
      # NLL of combined log-probs against actual symbol, masked.
      nll_per = -combined_log_probs.gather(1, symbols_t.unsqueeze(1)).squeeze(1)
      total_loss = total_loss + (nll_per * mask_t).sum()

    total_loss.backward()
    # Per-component clip budgets: LSTM keeps torch_compress's 4.0; the
    # Transformer-XL half uses NNCP-aligned clip_xl (0.25); the mixer is
    # tiny so a generous 4.0 is fine.
    _clip_xl = float(globals().get('clip_xl', 0.25))
    torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 4.0)
    torch.nn.utils.clip_grad_norm_(model.transformer_xl.parameters(), _clip_xl)
    if model.mixer is not None:
      torch.nn.utils.clip_grad_norm_(model.mixer.parameters(), 4.0)
    lstm_opt.step()
    xl_opt.step()
    if mixer_opt is not None:
      mixer_opt.step()

    # ---- Compute ensemble cross-entropy for logging (against AC's distribution) ----
    # Use the *same* combined_log_probs the AC saw (detached) so the running
    # average matches the realised bpc the AC is producing.
    one_hot = F.one_hot(symbols_t, num_classes=vocab_size).float()
    loss_vals = -(one_hot * combined_log_probs.detach()).sum(dim=-1)
    cross_entropy += (loss_vals * mask_t).sum().item()
    denom += mask_t.sum().item()

    # ---- State carry ----
    # LSTM: detach + push to queue
    lstm_states_queue.append([(h.detach(), c.detach()) for (h, c) in new_lstm_state])
    # Transformer-XL: mems already detached inside _update_mems
    xl_mems = new_xl_mems

    # Slide both windows by the just-coded symbol
    symbols_unsq = symbols_t.unsqueeze(1)
    lstm_seq_input = torch.cat([lstm_seq_input[:, 1:], symbols_unsq], dim=1)
    xl_seq_input = torch.cat([xl_seq_input[:, 1:], symbols_unsq], dim=1)

    pos += 1

    if time.time() - last_print_time >= 20:
      last_print_time = time.time()
      time_diff = last_print_time - start
      current_bpc = (cross_entropy / denom) / np.log(2)
      print(template.format(
          pos / split * 100, current_bpc, time_diff,
          current_lstm_lr, current_xl_lr, pos))
      if tb_writer is not None:
        tb_writer.add_scalar('train/bpc', current_bpc, pos)
        tb_writer.add_scalar('train/cross_entropy_total', cross_entropy, pos)
        tb_writer.add_scalar('train/lstm_lr', current_lstm_lr, pos)
        tb_writer.add_scalar('train/xl_lr', current_xl_lr, pos)
        tb_writer.add_scalar('train/elapsed_sec', time_diff, pos)
        tb_writer.add_scalar('train/steps_per_sec', pos / time_diff if time_diff > 0 else 0.0, pos)

  if tb_writer is not None:
    final_time = time.time() - start
    if denom > 0:
      tb_writer.add_scalar('train/bpc', (cross_entropy / denom) / np.log(2), pos)
    tb_writer.add_scalar('train/elapsed_sec', final_time, pos)
    tb_writer.add_text('summary', (
        f"final_pos={pos} elapsed_sec={final_time:.2f} "
        f"final_bpc={(cross_entropy / denom) / np.log(2) if denom > 0 else float('nan'):.4f}"
    ))
    tb_writer.flush()
    tb_writer.close()


def _retrain_hybrid(*, model, lstm_retrain_opt, xl_retrain_opt, lstm_lr, xl_lr,
                    file_data, file_pos, split, batch_size_streaming,
                    retrain_block_len, retrain_seq_length,
                    retrain_batch_size_lstm,
                    retrain_tgt_len, retrain_mem_len,
                    stream_mem_len, vocab_size, device):
  """Retrain both submodels on the same trailing window in their native shapes.

  - LSTM: BPTT batches of (retrain_seq_length+1)-token sequences over the
    last ``retrain_block_len`` chars, sharded across the same parallel-stream
    layout the streaming loop uses (i.e. each stream's history is its own
    column). This mirrors the data-construction code embedded in the LSTM
    solo retrain block of ``process()`` exactly; if that body changes,
    update here.
  - Transformer-XL: NNCP-style streaming pass via ``_retrain_transformer_xl``,
    which itself reset_lengths the model to (retrain_tgt_len, 0,
    retrain_mem_len) and back. Each retrain pass on the trailing
    ``retrain_block_len`` chars (capped at retrain_block_len for our
    transformer's retrain_block_len which is independent of the LSTM one).

  Both retrains see the same recent data but in different shapes; each
  updates only its own submodel.
  """
  # ---- LSTM half ----
  pos = file_pos - 1  # streaming step at retrain trigger
  r_start_step = max(0, pos - retrain_block_len)
  r_end_step = pos

  all_inputs = []
  all_targets = []
  r_step = r_start_step
  while r_step < r_end_step:
    for i in range(batch_size_streaming):
      base_idx = r_step + i * split
      start_idx = base_idx
      end_idx = start_idx + retrain_seq_length + 1
      current_stream_limit = i * split + pos + 1
      stream_segment = file_data[start_idx: min(end_idx, current_stream_limit)]
      if len(stream_segment) < retrain_seq_length + 1:
        stream_segment = list(stream_segment) + [0] * (retrain_seq_length + 1 - len(stream_segment))
      all_inputs.append(stream_segment[:-1])
      all_targets.append(stream_segment[1:])
    r_step += retrain_seq_length

  if all_inputs:
    all_inputs_t = torch.tensor(all_inputs, dtype=torch.long, device=device)
    all_targets_t = torch.tensor(all_targets, dtype=torch.long, device=device)

    total_examples = all_inputs_t.shape[0]
    remainder = total_examples % retrain_batch_size_lstm
    if remainder != 0:
      all_inputs_t = all_inputs_t[:-remainder]
      all_targets_t = all_targets_t[:-remainder]
      total_examples -= remainder

    if total_examples > 0:
      for i in range(0, total_examples, retrain_batch_size_lstm):
        batch_inputs = all_inputs_t[i: i + retrain_batch_size_lstm]
        batch_targets = all_targets_t[i: i + retrain_batch_size_lstm]
        # Reuse the same retrain_step helper used by the LSTM solo path,
        # but with a single-element model list and the LSTM submodel.
        retrain_step([model.lstm], [lstm_retrain_opt], batch_inputs, batch_targets, lstm_lr)

  # ---- Transformer-XL half ----
  # Reuse the existing _retrain_transformer_xl helper exactly as the
  # transformer solo path does; pass [model.transformer_xl] as the
  # single-element models list.
  _retrain_transformer_xl(
      models=[model.transformer_xl],
      retrain_optimizers=[xl_retrain_opt],
      current_lr=xl_lr,
      file_data=file_data,
      file_pos=file_pos,
      retrain_block_len=retrain_block_len,
      retrain_tgt_len=retrain_tgt_len,
      retrain_mem_len=retrain_mem_len,
      retrain_batch_size=retrain_batch_size_lstm,
      stream_mem_len=stream_mem_len,
      vocab_size=vocab_size,
      device=device,
  )
'''


# ---------------------------------------------------------------------------
# Build the new notebook by mutating the original
# ---------------------------------------------------------------------------

with open(SRC_NB, 'r') as f:
    nb = json.load(f)

# Title markdown cell
nb['cells'][0] = md_cell("# torch-compress\n")

# Subtitle markdown — clarify this is a PyTorch port of the JAX version
nb['cells'][1] = md_cell(
    "PyTorch port of [jax-compress](https://github.com/byronknoll/jax-compress) "
    "by Byron Knoll. Same compression algorithm, neural network, and arithmetic "
    "coder as the JAX/Flax version; the model, training loop, and checkpoint "
    "format have been rewritten with `torch.nn`.\n"
)

# Append TensorBoard params to the existing Parameters cell.
PARAMS_TB_APPEND = '''
#@markdown ---
tensorboard = False #@param {type:"boolean"}
#@markdown _If enabled, logs training metrics (bpc, loss, lr, retrain loss) to TensorBoard during compression / decompression._

#@markdown ---
tensorboard_run_name = "torch_compress" #@param {type:"string"}
#@markdown _Subdirectory name for this run inside `tensorboard_logdir`. Use distinct names to compare runs in TensorBoard._

#@markdown ---
tensorboard_logdir = "data/tensorboard" #@param {type:"string"}
#@markdown _Parent directory for TensorBoard event files. Run `tensorboard --logdir <logdir>` to view all runs side by side._

#@markdown ---
use_bf16 = True #@param {type:"boolean"}
#@markdown _Run the model forward/backward in bfloat16 mixed precision. ~2x faster on bf16-capable GPUs (Ampere/Hopper, GH200/H100). Compress and decompress MUST use the same setting -- a file compressed with use_bf16=True can only be decompressed with use_bf16=True (and vice versa)._

#@markdown ---
use_fp16 = False #@param {type:"boolean"}
#@markdown _Run the model in float16 mixed precision. NNCP-style. Takes precedence over `use_bf16`. fp16 has more mantissa precision than bf16 (better for deep transformer accumulation) but a much smaller exponent range -- gradients can underflow. We do NOT use `GradScaler`, so this is best-effort: untested for round-trip stability and may produce different bpc than bf16 even on the same input. Round-trip safety must be verified per run via `--mode both`._

#@markdown ---
model_type = "lstm" #@param ["lstm", "transformer_xl"]
#@markdown _Predictor architecture. "lstm" is the default reference model. "transformer_xl" routes through the NNCP-style adapter (`models.transformer_xl.TransformerXLModel`) and requires the local repo (the `models/` package must be importable -- not available in a vanilla Colab clone-less session)._

#@markdown ---
#@markdown ### Transformer-XL hparams
#@markdown _Used when `model_type == "transformer_xl"` (XL-solo) or as the XL submodel in `model_type == "hybrid"`. Defaults now track NNCP v2's `nncp_enwik_large.sh` config (d_model=1024, d_head=128, d_inner=4096). The smaller `nncp_enwik_base.sh` config (d_model=512, d_head=64, d_inner=2048) was the previous default. Empirically the larger XL submodel improved enwik8 hybrid+mixer by 0.0099 bpc (rb10m 1.2587 -> xl_large 1.2488) at ~5.9x wall cost (14h45m -> 86h31m). Round-trip md5-verified on enwik4 dry run with same large hparams._
n_layer = 12 #@param {type:"integer"}
n_head = 8 #@param {type:"integer"}
d_model = 1024 #@param {type:"integer"}
d_head = 128 #@param {type:"integer"}
d_inner = 4096 #@param {type:"integer"}
mem_len = 160 #@param {type:"integer"}
ext_tgt_len = 31 #@param {type:"integer"}
attn_type = 1 #@param {type:"integer"}
tied_r_bias = True #@param {type:"boolean"}
use_gelu = True #@param {type:"boolean"}
dropout = 0.25 #@param {type:"number"}
dropatt = 0.0 #@param {type:"number"}
init_std = 0.013 #@param {type:"number"}
retrain_tgt_len = 64 #@param {type:"integer"}
retrain_mem_len = 128 #@param {type:"integer"}
#@markdown _The model is constructed with `dropout`, but streaming forward passes use `deterministic=True` (eval mode, dropout off). Dropout activates only during retraining (`deterministic=False`, train mode). `retrain_tgt_len` and `retrain_mem_len` are NNCP's retrain-time shape — `model.reset_length()` swaps these in around each retrain pass and restores the streaming shape (1, 0, mem_len) afterward._

#@markdown ---
#@markdown _Transformer-XL learning-rate schedule. Defaults now track NNCP's `nncp_enwik_large.sh`: starting LR 4.0e-5 (was 7.9e-5 from base), to compensate for the larger model's higher gradient magnitude. The 341K/3.13M transitions don't fire at our enwik8 budget (~202K streaming steps) so training is effectively constant; we tried pulling the transitions in to 50K/150K (commit 7407e2f) but it regressed enwik8 by +0.0089 bpc and was reverted (the 5e-6 floor in the final ~25% of training undertrains late-document tokens). Retrain LR also lowered to match nncp_enwik_large._
learning_rate_schedule_xl = "0:4.0e-5 341105:1.3e-5 3134681:4.0e-6" #@param {type:"string"}
retrain_lr_schedule_xl = "0:1.6e-4 13000:1.6e-4 93000:7.9e-5 163000:4.0e-5 1911300:1.3e-5" #@param {type:"string"}

#@markdown ---
#@markdown ### Transformer-XL optimizer hparams (NNCP-aligned)
#@markdown _Used by `_process_transformer_xl` and the transformer half of `_process_hybrid`. NNCP-base uses `clip=0.25` (vs torch_compress's tuned 4.0 for the LSTM) and `adam_eps=1e-9` (vs torch_compress's 1e-12). Closer alignment with NNCP's published config._
clip_xl = 0.25 #@param {type:"number"}
adam_eps_xl = 1e-9 #@param {type:"number"}
'''
base_params = ''.join(nb['cells'][3]['source'])
# Patch retrain_block_len 100K -> 10M (NNCP-base value). Empirically beats
# 100K by 0.0039 bpc on enwik8 (mixer_v2 1.2626 -> mixer_rb10m 1.2587) at
# ~10% wall-clock cost. The original 100K was inherited from the LSTM-only
# notebook tuning; NNCP's transformer config has always used 10M.
base_params = base_params.replace(
    'retrain_block_len = 100000 #@param {type:"integer"}\n'
    '#@markdown _Retrain over the last M symbols._\n',
    'retrain_block_len = 10000000 #@param {type:"integer"}\n'
    '#@markdown _Retrain over the last M symbols. NNCP-base value (10M); '
    'updated from the original 100K based on enwik8 evidence -- '
    'mixer_v2 1.2626 bpc -> mixer_rb10m 1.2587 bpc (-0.0039) at '
    '~10% wall-clock cost. The bigger window does not increase per-retrain '
    'compute (retrain_batch_size * retrain_seq_length sets that); it only '
    'widens the trailing context retrain may sample from._\n',
)
# Patch batch_size 128 -> 64 (NNCP-large value). Required for VRAM headroom
# on the XL-large submodel (154M XL params + 142M LSTM = ~295M total at
# bf16). Halving batch doubles per-stream step count and lengthens wall.
base_params = base_params.replace(
    'batch_size = 128 #@param {type:"integer"}\n'
    '#@markdown _Splits the file into N batches to process them in parallel. Increasing this value improves speed but may negatively affect the compression rate. For optimal speed on certain GPUs, set this to a multiple of 8._\n',
    'batch_size = 64 #@param {type:"integer"}\n'
    '#@markdown _Splits the file into N parallel streams. NNCP-large value '
    '(64); the original 128 was LSTM-tuned and risks OOM on the XL-large '
    'submodel (154M params). Override to 128 for transformer_xl-base or '
    'LSTM-only configurations._\n',
)
# Patch retrain_batch_size 256 -> 32 (NNCP-large value). Smaller retrain
# batch reduces VRAM pressure during the retrain pass at the bigger XL.
base_params = base_params.replace(
    'retrain_batch_size = 256 #@param {type:"integer"}\n'
    '#@markdown _The batch size designated for retraining. Increasing this improves parallelism during the retraining phase._\n',
    'retrain_batch_size = 32 #@param {type:"integer"}\n'
    '#@markdown _Batch dim during retraining. NNCP-large value (32); the '
    'original 256 was LSTM-tuned. Smaller retrain batch is cheaper per pass '
    'and stays well under the streaming-batch VRAM peak._\n',
)
params_src = base_params + PARAMS_TB_APPEND
nb['cells'][3] = code_cell(params_src, tags=["parameters"])

# Replace JAX-heavy code cells
nb['cells'][5] = code_cell(IMPORTS_SRC)
nb['cells'][6] = code_cell(SYSTEM_INFO_SRC)
nb['cells'][9] = code_cell(MODEL_ARCH_SRC)
nb['cells'][10] = code_cell(COMPRESSION_LIB_SRC)

# Update stale flax comment in the "Download Result" cell (cell 15).
download_src = ''.join(nb['cells'][15]['source'])
download_src = download_src.replace(
    '# flax.training.checkpoints creates files like "checkpoint_0"\n'
    '    # We should zip or download the directory\n',
    '# Bundle the data/ckpt directory (model weights, optimizer state, AC state)\n'
    '    # into a single zip for download.\n',
)
nb['cells'][15] = code_cell(download_src)

# Validate cell indices: ensure no remaining 'jax' / 'flax' / 'optax' references
import re
JAX_PATTERN = re.compile(r"\b(jax|jnp|flax|optax)\b")

issues = []
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if JAX_PATTERN.search(src):
        issues.append((idx, [m.group() for m in JAX_PATTERN.finditer(src)]))

if issues:
    print("WARNING: Remaining JAX-style references in cells:")
    for idx, hits in issues:
        print(f"  cell {idx}: {hits}")
else:
    print("OK: no JAX references remain.")

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {DST_NB}")
print(f"Cell count: {len(nb['cells'])}")


# ---------------------------------------------------------------------------
# Slim decompressor notebook (torch_compress_decompressor.ipynb)
# ---------------------------------------------------------------------------
# Goal: smallest LTCB-compatible notebook that reads compressed.dat (produced by
# torch_compress.ipynb) and reproduces the original. Hardcoded params, no UI,
# no compression code path, no encoder/output-stream classes.

SLIM_PARAMS_SRC = '''# Decompressor parameters (must match the values used during compression).
batch_size = 128
seq_length = 15
rnn_units = 1400
num_layers = 8
embedding_size = 512
ensemble_size = 1
learning_rate_schedule = "0:0.0005 200000:0.0002"
adam_b1 = 0.0
adam_b2 = 0.9999
adam_eps = 1e-12
mode = "decompress"
preprocess = "nncp"
n_words = 8192
min_freq = 64
path_to_file = "data/compressed.dat"
checkpoint = False
total_parts = 1
current_part = 1
retrain_period_schedule = "0:1001 200000:5001"
retrain_block_len = 100000
retrain_seq_length = 100
retrain_batch_size = 256
retrain_lr_schedule = "0:0.0005 200000:0.0002"
retrain_dropout = 0.4
download_option = "no_download"
model_type = "lstm"
'''


SLIM_AC_SRC = '''# Reference arithmetic decoding (decoder + bit input only).
# Source: https://www.nayuki.io/page/reference-arithmetic-coding (MIT)

class _ACBase:
  def __init__(self, numbits):
    self.num_state_bits = numbits
    self.full_range = 1 << numbits
    self.half_range = self.full_range >> 1
    self.quarter_range = self.half_range >> 1
    self.minimum_range = self.quarter_range + 2
    self.maximum_total = self.minimum_range
    self.state_mask = self.full_range - 1
    self.low = 0
    self.high = self.state_mask

  def update(self, freqs, symbol):
    low, high = self.low, self.high
    rng = high - low + 1
    total = int(freqs[-1])
    symlow = int(freqs[symbol-1]) if symbol > 0 else 0
    symhigh = int(freqs[symbol])
    self.low = low + symlow * rng // total
    self.high = low + symhigh * rng // total - 1
    while ((self.low ^ self.high) & self.half_range) == 0:
      self.shift()
      self.low = (self.low << 1) & self.state_mask
      self.high = ((self.high << 1) & self.state_mask) | 1
    while (self.low & ~self.high & self.quarter_range) != 0:
      self.underflow()
      self.low = (self.low << 1) ^ self.half_range
      self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1


class ArithmeticDecoder(_ACBase):
  def __init__(self, numbits, bitin):
    super().__init__(numbits)
    self.input = bitin
    self.code = 0
    for _ in range(self.num_state_bits):
      self.code = self.code << 1 | self._read_bit()

  def read(self, freqs):
    total = int(freqs[-1])
    rng = self.high - self.low + 1
    offset = self.code - self.low
    value = ((offset + 1) * total - 1) // rng
    start, end = 0, len(freqs)
    while end - start > 1:
      mid = (start + end) >> 1
      lo = int(freqs[mid-1]) if mid > 0 else 0
      if lo > value:
        end = mid
      else:
        start = mid
    self.update(freqs, start)
    return start

  def shift(self):
    self.code = ((self.code << 1) & self.state_mask) | self._read_bit()

  def underflow(self):
    self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self._read_bit()

  def _read_bit(self):
    b = self.input.read()
    return 0 if b == -1 else b


class BitInputStream:
  def __init__(self, inp):
    self.input = inp
    self.currentbyte = 0
    self.numbitsremaining = 0

  def read(self):
    if self.currentbyte == -1:
      return -1
    if self.numbitsremaining == 0:
      b = self.input.read(1)
      if not b:
        self.currentbyte = -1
        return -1
      self.currentbyte = b[0]
      self.numbitsremaining = 8
    self.numbitsremaining -= 1
    return (self.currentbyte >> self.numbitsremaining) & 1

  def close(self):
    self.input.close()
    self.currentbyte = -1
    self.numbitsremaining = 0
'''


SLIM_DECOMPRESS_SRC = '''#@title Decompression

# This slim notebook only handles the single-shot case. For multi-part /
# resumable decompression use the full torch_compress.ipynb instead.
assert current_part == 1 and total_parts == 1, (
    "Slim decompressor only supports current_part=1, total_parts=1. "
    "Use torch_compress.ipynb for multi-part decompression."
)

# Build vocabulary size from the dictionary file (NNCP) or read it from the header.
if preprocess in ("nncp", "nncp-done"):
  vocab_size = int(subprocess.check_output(['wc', '-l', 'data/dictionary.words']).split()[0])
else:
  vocab_size = None  # Read from the 256-bit header.

with open(path_to_file, "rb") as inp:
  # 5-byte big-endian length header, then the bitstream payload follows directly.
  length = int.from_bytes(inp.read(5), byteorder='big')
  bitin = BitInputStream(inp)

  if vocab_size is None:
    vocab = []
    for i in range(256):
      if bitin.read():
        vocab.append(i)
    vocab_size = len(vocab)
  else:
    vocab = None

  vocab_size = math.ceil(vocab_size / 8) * 8
  output = [0] * length
  dec = ArithmeticDecoder(32, bitin)

  process(False, length, vocab_size, dec, output)

  output_path = "data/decompressed.dat"
  with open(output_path, "wb") as out:
    if preprocess in ("nncp", "nncp-done"):
      for i in range(length):
        out.write(bytes((output[i] // 256,)))
        out.write(bytes((output[i] % 256,)))
    else:
      idx2char = np.array(vocab)
      for i in range(length):
        out.write(bytes((idx2char[output[i]],)))

# NNCP post-processing: invert the dictionary tokenization to recover the
# original bytes. (For preprocess='nncp-done' the user supplied an already-
# preprocessed file, so no inverse is run.)
if preprocess == "nncp":
  final_path = "data/final.dat"
  subprocess.run(
      ['./nncp/preprocess', 'd', 'data/dictionary.words', output_path, final_path],
      check=True,
  )
  output_path = final_path

print("Done. Output:", output_path)
'''


SLIM_MD5_SRC = '''#@title Verify

import hashlib

def _md5(path):
  h = hashlib.md5()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()

print(f"md5({output_path}) = {_md5(output_path)}")
'''


# Build the slim notebook from scratch
slim_nb = {
    "cells": [
        md_cell("# torch-compress (decompressor)\n\n"
                "Slim decompressor for files produced by `torch_compress.ipynb`. "
                "Hardcoded parameters; no compression path; no encoder/output-stream classes.\n"),
        code_cell(SLIM_PARAMS_SRC),
        code_cell(IMPORTS_SRC),       # Reuse: includes the determinism preamble
        code_cell(MODEL_ARCH_SRC),    # Reuse the same model
        code_cell(COMPRESSION_LIB_SRC),  # Reuse: process() handles both directions
        code_cell(SLIM_AC_SRC),
        code_cell(SLIM_DECOMPRESS_SRC),
        code_cell(SLIM_MD5_SRC),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

SLIM_DST = os.path.join(REPO_DIR, "torch_compress_decompressor.ipynb")
with open(SLIM_DST, 'w') as f:
    json.dump(slim_nb, f, indent=1)

print(f"Wrote {SLIM_DST}")
print(f"Slim cell count: {len(slim_nb['cells'])}")
