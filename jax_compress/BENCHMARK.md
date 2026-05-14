# enwik8 benchmark: JAX vs PyTorch port

Both runs use the same compression algorithm (LSTM + arithmetic coder + NNCP preprocessor) and the same hyperparameters. The PyTorch port replaces JAX/Flax/Optax with `torch.nn` and uses a deterministic `nn.LSTMCell` Python loop instead of cuDNN's fused RNN (required for LTCB-style determinism — see [README.md](README.md)).

## Headline numbers

| Metric                            | JAX (reference)            | PyTorch port              | Δ (torch − jax) |
|-----------------------------------|---------------------------:|--------------------------:|----------------:|
| Compressed size (bytes)           | 15,505,441                 | 16,148,077                | +642,636        |
| bpc                               | 1.2404                     | 1.2918                    | +0.0514         |
| Wall clock (s)                    | 13,707.98 (≈ 3.81 h)       | ≈ 52,200 (≈ 14.5 h)       | ≈ +38,500 (+10.7 h) |
| Hardware                          | TPU v6e-1                  | NVIDIA GH200 480 GB       | (n/a)           |
| md5(decompress(compressed)) match | yes                        | round-trip verified on enwik4 smoke test (10 KB); enwik8 decompression interrupted before completion | (must be ✓) |
| Original input                    | enwik8 (100,000,000 B)     | enwik8 (100,000,000 B)    |                 |

The bpc and compressed-size comparisons are the **real** measure of port fidelity. Wall-clock comparison is hardware-dependent and informative only as a rough cost estimate.

## Run config (must match on both sides)

| Param                       | Value                                  |
|-----------------------------|----------------------------------------|
| `batch_size`                | 128                                    |
| `seq_length`                | 15                                     |
| `rnn_units`                 | 1400                                   |
| `num_layers`                | 8                                      |
| `embedding_size`            | 512                                    |
| `ensemble_size`             | 1                                      |
| `learning_rate_schedule`    | `0:0.0005 200000:0.0002`               |
| Adam (b1, b2, eps)          | (0.0, 0.9999, 1e-12)                   |
| `preprocess`                | NNCP, `n_words=8192`, `min_freq=64`    |
| `retrain_period_schedule`   | `0:1001 200000:5001`                   |
| `retrain_block_len`         | 100,000                                |
| `retrain_seq_length`        | 100                                    |
| `retrain_batch_size`        | 256                                    |
| `retrain_lr_schedule`       | `0:0.0005 200000:0.0002`               |
| `retrain_dropout`           | 0.4                                    |

## TensorBoard view

Both runs land in the same logdir so they overlay on identical axes:

```
runs/
├── torch_enwik8/             # live torch run (writes train/bpc, retrain/*, ...)
└── baseline_jax_enwik8/      # static reference, written by scripts/log_jax_baseline.py
```

```bash
# After the torch run finishes, stamp the JAX baseline at the same final step
# so the reference line spans the full torch curve:
python scripts/log_jax_baseline.py --logdir runs --total-steps <torch_final_step>

# View
tensorboard --logdir runs
```

`baseline_jax_enwik8` appears as a flat line at bpc 1.2404; the live torch curve should converge toward (or below) it.

## Caveats

- **Hardware mismatch**: TPU v6e-1 vs the GPU(s) the torch run used. Compression ratio comparison is fair; wall-clock is not.
- The deterministic `nn.LSTMCell` Python loop in the PyTorch port loses cuDNN's fused-RNN speedup. Expected slowdown vs raw cuDNN is roughly 5-10×; the LTCB determinism requirement makes this trade unavoidable for a submission-shaped artifact.
- A meaningful gap (say |Δ bpc| > 0.05) would indicate implementation drift worth investigating: weight-init, dropout RNG stream, or NNCP preprocessing differences are the usual suspects.

## Compression-rate vs file-size sweep (PyTorch port)

To characterize how well the algorithm scales with input size, we compress prefixes of enwik8 with identical hyperparameters. Each row is an independent run starting from random weights.

| Size       | File   | Bytes        | Compressed (B) | bpc     | Wall clock | Precision | Notes                                    |
|-----------:|--------|-------------:|---------------:|--------:|-----------:|-----------|------------------------------------------|
| 10 KB      | enwik4 | 10,000       | 4,586          | 3.6688  | < 1 s      | fp32      | `preprocess=none` (NNCP needs more text) |
| 100 KB     | enwik5 | 100,000      | 33,331         | 2.6665  | 57.6 s     | bf16      | `preprocess=nncp`                        |
| 100 KB     | enwik5 | 100,000      | 33,331         | 2.6665  | 2.1 min    | fp32      | `preprocess=nncp`, md5 round-trip ✓      |
| 1 MB       | enwik6 | 1,000,000    | 249,902        | 1.9992  | _TBD_      | bf16      | `preprocess=nncp`                        |
| 1 MB       | enwik6 | 1,000,000    | 249,902        | 1.9992  | 13.1 min   | fp32      | `preprocess=nncp`, md5 round-trip ✓      |
| 10 MB      | enwik7 | 10,000,000   | 2,021,773      | 1.6174  | _TBD_      | bf16      | `preprocess=nncp`                        |
| 10 MB      | enwik7 | 10,000,000   | 2,019,901      | 1.6159  | 58.3 min   | fp32      | `preprocess=nncp`, compress-only         |
| 100 MB     | enwik8 | 100,000,000  | 16,148,077     | 1.2918  | ≈ 14.5 h   | fp32      | `preprocess=nncp`                        |

Expected behavior: bpc decreases monotonically with size, with the steepest drop in the 100 KB → 10 MB range as the model exits the random-init burn-in regime. Asymptotic per-byte cost (large N) is bounded below by the source's true entropy (≈ 0.9 bpc for English Wikipedia).
