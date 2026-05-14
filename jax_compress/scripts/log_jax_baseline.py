"""Stage the published JAX baseline numbers as TensorBoard scalars.

Writes the headline numbers from the README's "Benchmarks" section as scalars
under a synthetic run, so that when you do `tensorboard --logdir <logdir>` the
JAX result appears as a flat reference line on the same plot as the live torch
curve.

Usage:
  python scripts/log_jax_baseline.py --logdir runs
  # default --target enwik8, --run-name baseline_jax_enwik8

  # Make the reference line span the full torch run by passing the final step:
  python scripts/log_jax_baseline.py --logdir runs --total-steps 781250

The script does not modify or read any state outside its own --logdir, so it is
safe to invoke while a torch_compress.ipynb run is in progress as long as you
point --logdir somewhere distinct from the live run's logdir, OR use the same
parent dir but a unique --run-name (recommended — that's the whole point).
"""

import argparse
import os


# Numbers from jax_compress/README.md "Benchmarks" section.
JAX_RESULTS = {
    "enwik8": {
        "original_bytes": 100_000_000,
        "compressed_bytes": 15_505_441,
        "wall_clock_sec": 13_707.98,
        "hardware": "TPU v6e-1",
    },
    "enwik9": {
        "original_bytes": 1_000_000_000,
        "compressed_bytes": 113_393_442,
        "wall_clock_sec": 110_013.19,
        "hardware": "TPU v6e-1",
        "dictionary_bytes": 80_040,
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=list(JAX_RESULTS), default="enwik8")
    ap.add_argument("--logdir", default="runs",
                    help="parent tensorboard log directory (must match the torch run's logdir "
                         "for the reference line to overlay)")
    ap.add_argument("--run-name", default=None,
                    help="subdirectory under --logdir; defaults to baseline_jax_<target>")
    ap.add_argument("--total-steps", type=int, default=1,
                    help="last step at which to stamp the baseline scalar. "
                         "Set this to your torch run's final step to draw a flat line "
                         "spanning the full plot.")
    args = ap.parse_args()

    target = JAX_RESULTS[args.target]
    run_name = args.run_name or f"baseline_jax_{args.target}"
    out_dir = os.path.join(args.logdir, run_name)

    bpc = 8.0 * target["compressed_bytes"] / target["original_bytes"]

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=out_dir)

    # Two points at step 0 and step N produce a flat horizontal line.
    for step in (0, args.total_steps):
        writer.add_scalar("train/bpc", bpc, step)

    writer.add_scalar("train/elapsed_sec", target["wall_clock_sec"], args.total_steps)

    summary_lines = [
        f"backend=jax target={args.target} hardware={target['hardware']}",
        f"original_bytes={target['original_bytes']:,}",
        f"compressed_bytes={target['compressed_bytes']:,}",
        f"bpc={bpc:.4f}",
        f"wall_clock_sec={target['wall_clock_sec']:.2f}",
    ]
    if "dictionary_bytes" in target:
        summary_lines.append(f"dictionary_bytes={target['dictionary_bytes']:,}")
    writer.add_text("baseline/info", "\n".join(summary_lines))

    writer.flush()
    writer.close()
    print(f"wrote {out_dir}: bpc={bpc:.4f}, "
          f"wall_clock={target['wall_clock_sec']:.0f}s, "
          f"line spans steps [0, {args.total_steps}]")


if __name__ == "__main__":
    main()
