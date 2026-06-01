"""cuDNN nn.LSTM determinism acceptance test.

Six sub-tests, in order of severity. If Test 1 fails, cuDNN LSTM is unusable
for our compression pipeline (the arithmetic coder requires bit-identical
forward at every step). If Test 1 passes but Test 4 fails, cuDNN's backward
is non-deterministic (the known issue per PyTorch docs); we could still use
cuDNN for streaming forward but would need a fallback for retrain.

Configuration matches our LSTM submodel:
  input_size=512, hidden_size=1400, num_layers=8, batch=64, seq=15

We do NOT set torch.use_deterministic_algorithms(True) here -- the whole
point is to see whether cuDNN's fused LSTM gives us determinism we can rely
on, in the default mode it would actually run in.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# Match our LSTM submodel config.
INPUT_SIZE = 512
HIDDEN_SIZE = 1400
NUM_LAYERS = 8
BATCH = 64
SEQ = 15

DEVICE = torch.device('cuda')


def _seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_lstm():
    return nn.LSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=True,
    ).to(DEVICE)


def _make_input(seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(BATCH, SEQ, INPUT_SIZE, generator=g, device=DEVICE)
    h0 = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_SIZE, device=DEVICE)
    c0 = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_SIZE, device=DEVICE)
    return x, (h0, c0)


def _make_nonzero_state(seed=1):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    h0 = torch.randn(NUM_LAYERS, BATCH, HIDDEN_SIZE, generator=g, device=DEVICE)
    c0 = torch.randn(NUM_LAYERS, BATCH, HIDDEN_SIZE, generator=g, device=DEVICE)
    return h0, c0


def _report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
    return ok


def test_forward_repeat():
    """Test 1: forward twice with zero hidden state -> bit-identical?"""
    print("\nTest 1: forward repeatability, zero state")
    _seed_all(42)
    lstm = _make_lstm()
    x, hc = _make_input()
    with torch.no_grad():
        o1, (h1, c1) = lstm(x, hc)
        o2, (h2, c2) = lstm(x, hc)
    ok = (torch.equal(o1, o2) and torch.equal(h1, h2) and torch.equal(c1, c2))
    return _report("forward repeat (zero state)", ok,
                   detail=f"max|out diff|={(o1 - o2).abs().max().item():.2e}")


def test_forward_repeat_nonzero():
    """Test 2: forward twice with non-zero hidden state."""
    print("\nTest 2: forward repeatability, non-zero state")
    _seed_all(42)
    lstm = _make_lstm()
    x, _ = _make_input()
    h0, c0 = _make_nonzero_state()
    with torch.no_grad():
        o1, (h1, c1) = lstm(x, (h0, c0))
        o2, (h2, c2) = lstm(x, (h0, c0))
    ok = (torch.equal(o1, o2) and torch.equal(h1, h2) and torch.equal(c1, c2))
    return _report("forward repeat (non-zero state)", ok,
                   detail=f"max|out diff|={(o1 - o2).abs().max().item():.2e}")


def test_forward_bf16():
    """Test 3: forward twice under bf16 autocast."""
    print("\nTest 3: forward repeatability, bf16 autocast")
    _seed_all(42)
    lstm = _make_lstm()
    x, hc = _make_input()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        o1, (h1, c1) = lstm(x, hc)
        o2, (h2, c2) = lstm(x, hc)
    ok = (torch.equal(o1, o2) and torch.equal(h1, h2) and torch.equal(c1, c2))
    return _report("forward repeat (bf16)", ok,
                   detail=f"max|out diff|={(o1.float() - o2.float()).abs().max().item():.2e}")


def test_backward_repeat():
    """Test 4: backward twice from same forward -> gradients bit-identical?

    cuDNN's RNN backward uses thread-block atomics for gradient accumulation
    per PyTorch docs. We expect this to fail.
    """
    print("\nTest 4: backward repeatability (gradient bit-equality)")
    _seed_all(42)
    lstm = _make_lstm()
    x, hc = _make_input()
    # First backward.
    lstm.zero_grad()
    o1, _ = lstm(x, hc)
    o1.sum().backward()
    g1 = [p.grad.detach().clone() for p in lstm.parameters()]
    # Second backward with fresh state.
    lstm.zero_grad()
    o2, _ = lstm(x, hc)
    o2.sum().backward()
    g2 = [p.grad.detach().clone() for p in lstm.parameters()]
    all_eq = all(torch.equal(a, b) for a, b in zip(g1, g2))
    if all_eq:
        return _report("backward repeat (gradients bit-equal)", True)
    # Quantify how far off.
    max_diff = max((a - b).abs().max().item() for a, b in zip(g1, g2))
    return _report("backward repeat (gradients bit-equal)", False,
                   detail=f"max |grad diff|={max_diff:.2e}")


def test_train_then_forward_repeat():
    """Test 5: train one step, then forward twice on same input -> equal?

    Tests whether the optimizer-stepped model is itself deterministic in
    forward, even if the gradient that produced the step wasn't.
    """
    print("\nTest 5: forward repeatability after one optimizer step")
    _seed_all(42)
    lstm = _make_lstm()
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-4)
    x, hc = _make_input()
    # One training step.
    opt.zero_grad()
    o, _ = lstm(x, hc)
    o.sum().backward()
    opt.step()
    # Now two forwards on the same input -> equal?
    with torch.no_grad():
        o1, _ = lstm(x, hc)
        o2, _ = lstm(x, hc)
    ok = torch.equal(o1, o2)
    return _report("forward repeat post-step", ok,
                   detail=f"max|out diff|={(o1 - o2).abs().max().item():.2e}")


def test_cross_process():
    """Test 6: two subprocess invocations of full train cycle produce equal weights?

    Spawns this same script with --child to run a fixed init+train cycle
    and dump weights. Compares the two outputs.
    """
    print("\nTest 6: cross-process training determinism")
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, 'a.pt')
        b = os.path.join(d, 'b.pt')
        for path in (a, b):
            r = subprocess.run(
                [sys.executable, __file__, "--child", "--out", path],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(r.stdout); print(r.stderr, file=sys.stderr)
                return _report("cross-process train", False,
                               detail="child failed")
        sa, sb = torch.load(a), torch.load(b)
        keys = list(sa.keys())
        all_eq = all(torch.equal(sa[k], sb[k]) for k in keys)
        if all_eq:
            return _report("cross-process train", True)
        max_diff = max((sa[k] - sb[k]).abs().max().item() for k in keys)
        return _report("cross-process train", False,
                       detail=f"max |weight diff|={max_diff:.2e}")


def _child(out_path):
    """Subprocess body for Test 6: fixed init + train + dump weights."""
    _seed_all(42)
    lstm = _make_lstm()
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-4)
    x, hc = _make_input()
    for _ in range(20):
        opt.zero_grad()
        o, _ = lstm(x, hc)
        o.sum().backward()
        opt.step()
    state = {k: v.detach().clone() for k, v in lstm.state_dict().items()}
    torch.save(state, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.child:
        _child(args.out)
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN deterministic flag: {torch.backends.cudnn.deterministic}")
    print(f"use_deterministic_algorithms: {torch.are_deterministic_algorithms_enabled()}")
    print(f"\nLSTM config: input={INPUT_SIZE}, hidden={HIDDEN_SIZE}, "
          f"layers={NUM_LAYERS}, batch={BATCH}, seq={SEQ}")
    print()

    results = [
        test_forward_repeat(),
        test_forward_repeat_nonzero(),
        test_forward_bf16(),
        test_backward_repeat(),
        test_train_then_forward_repeat(),
        test_cross_process(),
    ]

    print(f"\n{'='*60}")
    n_pass = sum(results)
    print(f"Total: {n_pass}/6 passed")
    if results == [True, True, True, True, True, True]:
        print("VERDICT: cuDNN nn.LSTM fully usable. Refactor is viable.")
    elif results[:3] == [True, True, True] and not results[3]:
        print("VERDICT: forward is deterministic but backward is NOT. "
              "Could use cuDNN for streaming forward + LSTMCell for retrain.")
    elif not results[0]:
        print("VERDICT: cuDNN nn.LSTM fails even basic forward repeat. "
              "Unusable. Look at torch.compile for the LSTMCell loop instead.")
    else:
        print("VERDICT: partial pass. Mixed picture; analyze per-test.")


if __name__ == "__main__":
    main()
