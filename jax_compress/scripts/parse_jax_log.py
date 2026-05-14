"""Convert the stdout of jax_compress.ipynb into TensorBoard scalar events.

The JAX notebook prints lines like:

    {pct}%\\tcross entropy: {bpc}\\ttime: {time}\\tlr: {lr}\\tstep: {pos}

and (during retraining):

    Starting retraining at step {pos}... (period={p}, lr={lr}, examples={n})
    Retraining finished. Duration: {sec}s

This script regex-parses both, then writes the same scalar names that
torch_compress.ipynb uses (`train/bpc`, `train/lr`, `train/elapsed_sec`,
`train/steps_per_sec`, `retrain/lr`, `retrain/duration_sec`) so the JAX and
PyTorch runs can be compared on identical axes in TensorBoard.

Usage:
  python scripts/parse_jax_log.py jax_run.log --run-name jax_v1
  tensorboard --logdir data/tensorboard
"""

import argparse
import os
import re
import sys


# Sample line:
#   42.34%\tcross entropy: 1.23\ttime: 87.65\tlr: 0.00050000\tstep: 12345
PRINT_RE = re.compile(
    r'(?P<pct>\d+\.\d+)%\s+cross entropy:\s+(?P<bpc>-?\d+\.\d+)\s+'
    r'time:\s+(?P<sec>\d+\.\d+)\s+lr:\s+(?P<lr>[\d.eE+-]+)\s+step:\s+(?P<step>\d+)'
)

RETRAIN_START_RE = re.compile(
    r'Starting retraining at step (?P<step>\d+)\.\.\.\s*\(period=(?P<period>\d+\.\d+),\s+'
    r'lr=(?P<lr>[\d.eE+-]+),\s+examples=(?P<n>\d+)\)'
)

RETRAIN_END_RE = re.compile(
    r'Retraining finished\. Duration:\s+(?P<sec>\d+\.\d+)s'
)


def parse(log_path, writer):
    pending_retrain = None  # holds dict from a START line until matching END
    n_train = 0
    n_retrain = 0

    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            m = PRINT_RE.search(line)
            if m:
                step = int(m['step'])
                bpc = float(m['bpc'])
                sec = float(m['sec'])
                lr = float(m['lr'])
                writer.add_scalar('train/bpc', bpc, step)
                writer.add_scalar('train/lr', lr, step)
                writer.add_scalar('train/elapsed_sec', sec, step)
                if sec > 0:
                    writer.add_scalar('train/steps_per_sec', step / sec, step)
                n_train += 1
                continue

            m = RETRAIN_START_RE.search(line)
            if m:
                pending_retrain = {
                    'step': int(m['step']),
                    'lr': float(m['lr']),
                    'examples': int(m['n']),
                }
                continue

            m = RETRAIN_END_RE.search(line)
            if m and pending_retrain is not None:
                step = pending_retrain['step']
                writer.add_scalar('retrain/lr', pending_retrain['lr'], step)
                writer.add_scalar('retrain/examples', pending_retrain['examples'], step)
                writer.add_scalar('retrain/duration_sec', float(m['sec']), step)
                pending_retrain = None
                n_retrain += 1

    return n_train, n_retrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', help="Path to a saved JAX (or PyTorch) notebook stdout file.")
    ap.add_argument('--run-name', default='jax_compress',
                    help="subdirectory name under --logdir (use distinct names per run)")
    ap.add_argument('--logdir', default='data/tensorboard',
                    help="parent directory for tensorboard event files")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        sys.exit(2)

    # Imported lazily so the module also serves as a quick docstring reference
    # without forcing torch as a dependency at module-import time.
    from torch.utils.tensorboard import SummaryWriter

    out_dir = os.path.join(args.logdir, args.run_name)
    writer = SummaryWriter(log_dir=out_dir)
    writer.add_text('config', f"backend=jax source_log={args.log}")

    n_train, n_retrain = parse(args.log, writer)
    writer.flush()
    writer.close()

    print(f"wrote {n_train} training points and {n_retrain} retrain events to {out_dir}")
    if n_train == 0:
        print("WARNING: no training lines matched the regex. "
              "Check that the log was produced by jax_compress.ipynb / torch_compress.ipynb "
              "and that you saved the notebook's full stdout.", file=sys.stderr)


if __name__ == '__main__':
    main()
