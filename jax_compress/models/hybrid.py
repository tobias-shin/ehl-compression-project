"""Container for an LSTM + Transformer-XL ensemble.

Holds both submodels via constructor injection so that ``.parameters()``,
``.to(device)``, optimizer wrapping, and parameter counting all work cleanly
through a single ``HybridModel`` reference. The actual ensemble logic
(per-step combining, AC interface, per-model backward, retrain) lives in
``_process_hybrid`` / ``_retrain_hybrid`` in the notebook, since it needs to
coordinate two distinct state-management patterns:

  - LSTM: ``(h, c)`` tuples in a ``states_queue`` of size ``seq_length``,
    with BPTT replay through the queue at each forward.
  - Transformer-XL: ``mems`` carry forward step-to-step; no replay.

The HybridModel itself only enforces the contract that both submodels share
a vocab_size (so the AC sees a single coherent distribution) -- it does not
itself implement a unified forward(). Use ``model.lstm`` and
``model.transformer_xl`` directly from the calling loop to get each
submodel's native ``(forward, init_states)`` interface.
"""
import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, lstm, transformer_xl, transformer_xl_small=None, mixer=None):
        super().__init__()
        assert lstm.vocab_size == transformer_xl.vocab_size, (
            f"vocab_size mismatch: LSTM={lstm.vocab_size}, "
            f"Transformer-XL={transformer_xl.vocab_size}; submodels must "
            f"see the same vocabulary so the geometric-mean ensemble produces "
            f"a coherent probability distribution for the AC to encode against."
        )
        if transformer_xl_small is not None:
            assert transformer_xl_small.vocab_size == lstm.vocab_size, (
                f"vocab_size mismatch on transformer_xl_small="
                f"{transformer_xl_small.vocab_size} vs LSTM={lstm.vocab_size}"
            )
        self.lstm = lstm
        self.transformer_xl = transformer_xl
        # Optional 3rd submodel: a small Transformer-XL with shallower depth
        # and shorter mem_len. Tests whether a similar-architecture 3rd
        # component can give the mixer useful diversity. If it does, a
        # structurally different 3rd model (e.g., a 1D causal CNN) is the
        # natural follow-up.
        self.transformer_xl_small = transformer_xl_small
        # Optional learned mixer. When None, the calling loop
        # (_process_hybrid) falls back to equal-weight geometric mean. When
        # set, the loop calls mixer(log_probs_list) to produce the combined
        # distribution and includes mixer params in its optimizer.
        self.mixer = mixer
        self.vocab_size = lstm.vocab_size
