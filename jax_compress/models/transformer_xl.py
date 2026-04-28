"""Transformer-XL adapter that exposes the same surface as ``LSTMModel``.

The training/compression loop in ``torch_compress.ipynb`` calls the model with::

    inputs: (batch, seq_length) int64
    states: list of per-layer states (or None on the first call)
    -> (logits, new_states)
       logits:     (batch, vocab)            if not return_sequence
                   (batch, seq, vocab)        if return_sequence
       new_states: list of per-layer states

NNCP's ``MemTransformerLM`` instead expects::

    data: (qlen, bsz) int64                 # sequence-first
    tlen: int                               # how many of the trailing tokens get logits
    *mems: list of n_layer+1 cached tensors
    -> [output, *new_mems]
       output: (tlen, bsz, vocab)

This adapter is purely a shape/argument translator — no behaviour change. It
lets the existing training loop drop in a Transformer-XL backbone behind the
same factory-style interface used today for ``LSTMModel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mem_transformer import MemTransformerLM


class TransformerXLModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 12,
        n_head: int = 8,
        d_model: int = 512,
        d_head: int = 64,
        d_inner: int = 2048,
        dropout: float = 0.0,
        dropatt: float = 0.0,
        tgt_len: int = 1,
        ext_len: int = 0,
        mem_len: int = 160,
        attn_type: int = 1,
        tied_r_bias: bool = True,
        use_gelu: bool = True,
        pre_lnorm: bool = False,
        same_length: bool = False,
        clamp_len: int = -1,
        d_embed: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len

        self.lm = MemTransformerLM(
            n_token=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            d_model=d_model,
            d_head=d_head,
            d_inner=d_inner,
            dropout=dropout,
            dropatt=dropatt,
            d_embed=d_embed,
            pre_lnorm=pre_lnorm,
            tgt_len=tgt_len,
            ext_len=ext_len,
            mem_len=mem_len,
            same_length=same_length,
            attn_type=attn_type,
            clamp_len=clamp_len,
            tied_r_bias=tied_r_bias,
            use_gelu=use_gelu,
        )

    def init_states(self, batch_size: int, device):
        del batch_size  # mems are sequence-indexed, not batch-indexed
        mems = self.lm.init_mems()
        if mems is None:
            return None
        return [m.to(device) for m in mems]

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int):
        # NNCP toggles between streaming-inference shape (tgt_len=1, mem_len≈160)
        # and retraining shape (tgt_len=64, mem_len=128). Phase-2 work will use this.
        self.lm.reset_length(tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len)
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len

    def forward(
        self,
        inputs: torch.Tensor,
        states,
        return_sequence: bool = False,
        deterministic: bool = True,
    ):
        data = inputs.transpose(0, 1).contiguous()  # (batch, seq) -> (seq, batch)
        qlen = data.size(0)
        tlen = qlen if return_sequence else 1

        was_training = self.lm.training
        self.lm.train(mode=not deterministic)
        try:
            mems = states if states else []
            ret = self.lm(data, tlen, *mems)
        finally:
            self.lm.train(was_training)

        output = ret[0]
        new_mems = list(ret[1:]) if len(ret) > 1 else None

        if return_sequence:
            logits = output.transpose(0, 1).contiguous()  # (tlen, b, V) -> (b, tlen, V)
        else:
            logits = output.squeeze(0)  # (1, b, V) -> (b, V)

        return logits, new_mems


def _smoke_test():
    """CPU-only forward-pass smoke test. Does not touch CUDA."""
    torch.manual_seed(0)
    vocab_size = 64
    batch_size = 3
    seq_length = 5

    model = TransformerXLModel(
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        d_model=16,
        d_head=8,
        d_inner=32,
        dropout=0.0,
        dropatt=0.0,
        tgt_len=1,
        ext_len=0,
        mem_len=8,
        attn_type=1,
        tied_r_bias=True,
        use_gelu=True,
    )
    # MemTransformerLM allocates its relative-position params with torch.Tensor(),
    # which leaves them uninitialised — NNCP's training script fills them via a
    # weights_init pass. The smoke test mirrors that subset; everything else
    # (nn.Linear, nn.LayerNorm, nn.Embedding) is fine on PyTorch defaults.
    with torch.no_grad():
        for attr in ("r_emb", "r_w_bias", "r_bias", "r_r_bias"):
            if hasattr(model.lm, attr):
                torch.nn.init.normal_(getattr(model.lm, attr), mean=0.0, std=0.02)

    device = torch.device("cpu")
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    states = model.init_states(batch_size, device)

    # 1) last-token-only path (compression-time shape)
    logits, new_states = model(inputs, states, return_sequence=False, deterministic=True)
    assert logits.shape == (batch_size, vocab_size), logits.shape
    assert isinstance(new_states, list) and len(new_states) == model.n_layer + 1

    # 2) full-sequence path (BPTT / retraining shape)
    logits_seq, new_states2 = model(
        inputs, states, return_sequence=True, deterministic=True
    )
    assert logits_seq.shape == (batch_size, seq_length, vocab_size), logits_seq.shape

    # 3) determinism: same inputs + same states -> same logits
    logits_b, _ = model(inputs, states, return_sequence=False, deterministic=True)
    assert torch.equal(logits, logits_b), "deterministic forward pass not bit-identical"

    # 4) mems carry: feeding new_states back in changes the output (memory is in use)
    logits_c, _ = model(inputs, new_states, return_sequence=False, deterministic=True)
    assert not torch.equal(logits, logits_c), "memory carry had no effect"

    print("transformer_xl smoke test OK:")
    print(f"  last-token logits: {tuple(logits.shape)}")
    print(f"  full-seq logits:   {tuple(logits_seq.shape)}")
    print(f"  mems: {len(new_states)} tensors, "
          f"sizes {[tuple(m.shape) for m in new_states]}")


if __name__ == "__main__":
    _smoke_test()
