"""Learned context-dependent gating mixer for the hybrid ensemble.

Replaces the equal-weight geometric mean of the LSTM and Transformer-XL
distributions with a tiny MLP that produces softmax-normalised per-step,
per-batch-member weights. The combined log-prob distribution is a weighted
sum of the per-submodel ``log_softmax`` outputs, equivalent to a *weighted*
geometric mean of the underlying probability distributions.

Input features (per submodel):
  - Entropy of the predicted distribution H_i = -sum(p_i log p_i).
    Captures "this model is uncertain right now."
  - Max log-prob: log(max p_i). Captures "this model has a sharp prediction."

Both are scalar per (batch, submodel), so the mixer's input dim is
``n_models * 2``. Output dim is ``n_models`` (softmax over models).

Training: the mixer is included in the forward graph, so a single backward
through the AC ensemble loss flows gradient into the mixer (and back through
the submodels). With softmax-normalised weights the equal-weight geometric
mean is in the hypothesis class (constant 0.5/0.5 weights), so the mixer
can never be strictly worse than the equal-weight ensemble at convergence.

Round-trip safety: weights depend deterministically on the per-submodel
log-probs (which depend deterministically on submodel params + input).
Same seed + same submodel state -> same mixer weights -> same combined
distribution -> same AC bytes. The mixer's online training updates are
also deterministic for the same gradient sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedMixer(nn.Module):
    def __init__(self, n_models: int = 2, hidden_dim: int = 64, init_std: float = 0.02):
        super().__init__()
        self.n_models = n_models
        in_dim = n_models * 2  # entropy + max log-prob per submodel
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_models),
        )
        # Init so the gate starts close to uniform. Small linear weights +
        # zero biases means the first-layer pre-activations are tiny,
        # GELU(tiny) is tiny, second layer output is tiny, softmax(tiny) is
        # near-uniform. The equal-weight ensemble is therefore the
        # initialisation; training nudges away from it.
        with torch.no_grad():
            for m in self.gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=init_std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, log_probs_list):
        """
        Args:
            log_probs_list: list of length ``n_models``, each element a
                ``(batch, vocab)`` tensor of log-probabilities (already
                ``log_softmax``'d by the caller).

        Returns:
            combined_log_probs: ``(batch, vocab)`` tensor. NOT necessarily
                a normalised log-prob distribution (the weighted sum of
                normalised log-probs is generally unnormalised); the caller
                should ``log_softmax`` it again before passing to AC, or
                ``softmax`` to get probabilities. Mathematically this is the
                log of the unnormalised weighted geometric mean of the
                submodels' probability distributions.
        """
        # Per-submodel scalar features (entropy + max log-prob).
        feats = []
        for lp in log_probs_list:
            # entropy in nats: -sum(exp(lp) * lp)
            ent = -(lp.exp() * lp).sum(dim=-1)               # (B,)
            max_lp = lp.max(dim=-1).values                   # (B,)
            feats.append(torch.stack([ent, max_lp], dim=-1))  # (B, 2)
        x = torch.cat(feats, dim=-1)                         # (B, n_models*2)

        gate_logits = self.gate(x)                           # (B, n_models)
        weights = F.softmax(gate_logits, dim=-1)             # (B, n_models)

        stacked = torch.stack(log_probs_list, dim=1)         # (B, n_models, V)
        combined = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, V)
        return combined
