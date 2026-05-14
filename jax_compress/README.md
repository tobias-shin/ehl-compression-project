# jax-compress

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byronknoll/jax-compress/blob/main/jax_compress.ipynb)

Made by Byron Knoll. GitHub repository: https://github.com/byronknoll/jax-compress

### Description
This project started as a JAX/Flax port of [tensorflow-compress](https://github.com/byronknoll/tensorflow-compress).

jax-compress performs lossless data compression using neural networks. It can run on TPUs and GPUs with a large batch size to achieve substantial speed improvements. It is designed for Google Colab, making it easy to run directly through a web browser. You can select a file, perform compression (or decompression), and download the resulting file.

The neural network is trained from scratch during both compression and decompression, meaning the model weights do not need to be stored in the compressed file. Arithmetic coding is used to encode the model predictions.

Feel free to contact me at byron@byronknoll.com if you have any questions.

### Instructions

**Basic Usage:** Configure all the fields in the "Parameters" section and select `Runtime -> Run All`.

**Advanced Usage:** Save a copy of this notebook to your own Google Drive and modify the code as needed.

**PyTorch port:** `torch_compress.ipynb` and `torch_compress_decompressor.ipynb` (the LTCB-style slim decompressor) are generated from `jax_compress.ipynb` by `scripts/build_torch_notebooks.py`. After regenerating, run `python scripts/smoke_test.py` for a fast CPU round-trip + determinism + cross-notebook check.

**Local CLI driver:** to compress / decompress a real file outside Colab, use `scripts/run_local.py`:

```bash
# Round-trip self-check on any file (verifies md5(input) == md5(decompressed))
python scripts/run_local.py roundtrip <file>

# Or compress / decompress separately
python scripts/run_local.py compress <input> <output.zc>
python scripts/run_local.py decompress <output.zc> <recovered>
```

Defaults to a small CPU model (`rnn_units=64, num_layers=2`); pass `--rnn-units 1400 --num-layers 8` to match the full Colab config.

NNCP-style periodic retraining is on by default with `--retrain-period "0:1001 200000:5001"` (i.e. retrain every ~1k–5k steps over the trailing `--retrain-block-len` symbols). For files small enough that retrain wouldn't fire (e.g. < a few KB at default batch size) it's a no-op. Pass `--no-retrain` to disable, or override any of `--retrain-period / --retrain-block-len / --retrain-seq-length / --retrain-batch-size / --retrain-lr-schedule / --retrain-dropout` to tune.

**TensorBoard comparison (jax-compress vs torch-compress).** `torch_compress.ipynb` and `scripts/run_local.py` can write training scalars (`train/bpc`, `train/lr`, `train/elapsed_sec`, `train/steps_per_sec`, plus retrain events) to TensorBoard. The original `jax_compress.ipynb` is not modified; instead, save its stdout and feed it to `scripts/parse_jax_log.py` to emit the same scalar names from the JAX run.

```bash
# 1. Torch run — turn tensorboard on, give the run a name
python scripts/run_local.py compress enwik6 enwik6.zc \
    --tensorboard --tb-run-name torch_v1 --tb-logdir runs

# 2. JAX run — capture stdout, then convert it to tb events
#    (in Colab: !python jax_compress.ipynb 2>&1 | tee jax_v1.log
#     or save the cell output, then download the log)
python scripts/parse_jax_log.py jax_v1.log --run-name jax_v1 --logdir runs

# 3. Compare side-by-side
tensorboard --logdir runs
```

The same tags are used for both backends, so JAX and PyTorch curves overlay directly. The torch notebook also accepts the corresponding params (`tensorboard`, `tensorboard_run_name`, `tensorboard_logdir`) inside Colab.

### Related Projects
*   [tensorflow-compress](https://github.com/byronknoll/tensorflow-compress)
*   [NNCP](https://bellard.org/nncp/)
*   [lstm-compress](https://github.com/byronknoll/lstm-compress)
*   [cmix](http://www.byronknoll.com/cmix.html)

### Benchmarks
These benchmarks were performed using jax-compress v1 with the default parameter settings on a v6e-1 TPU. Compression time and decompression time are approximately the same.

*   **enwik8:** Compressed to 15,505,441 bytes in 13,707.98 seconds.
*   **enwik9:** Compressed to 113,393,442 bytes in 110,013.19 seconds (Dictionary size: 80,040 bytes). 
    * The preprocessed enwik9 file was split into two parts. 
    * The "checkpoint" option was used to save and load model weights between processing each part. 
    * Article ordering preprocessing (from [fx2-cmix](https://github.com/kaitz/fx2-cmix)) was used:

```bash
cd enwik9-preproc/
make
./enwik9-preproc c enwik9
# After decompression:
./enwik9-preproc d final.dat
```

  * enwik9 decompressor size is 60,872 bytes. It is a zip file which contains:
      * jax_compress notebook
      * enwik9 article reordering code. This doesn't include the new_article_order, which is only needed for compression.
      * NNCP preprocessor code
      * dictionary for enwik9


See the [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html) for more information about the test files and a comparison with other compression programs.

### Versions
* **v1** - Released March 15, 2026. Changes from tensorflow-compress:
  * **Retraining:** Similar to NNCP, the model is periodically retrained using previously processed data.
  * Fixed a bug in the NNCP preprocessor.
