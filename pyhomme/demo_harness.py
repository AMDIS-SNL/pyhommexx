#!/usr/bin/env python3
"""Demonstrator for the P1+P2 harness: one differentiable subcycle with StubBackend.

What it exercises:
- bootstrap_model -> read initial state -> forward_step -> loss.backward()
- StubBackend.jtv returning zeros (validates the autograd wiring, NOT gradient correctness)
- MetricsCollector recording wall time, peak RSS, boundary-crossing bytes
- State pytree round-trip (write pre, read post)

What it does NOT exercise:
- Numerically correct gradients (StubBackend returns zeros by design; P4 FD + B3 JtV land later)
- Multi-step BPTT (that's M3 territory; this is a single-step smoke test)

Requires PYHOMMEXX_LIB_PATH env var pointing at the build dir with pyhommexx.*.so.
"""

import argparse
import os
import sys

import torch

sys.path.append(os.environ["PYHOMMEXX_LIB_PATH"])
from mpi4py import MPI  # noqa: E402,F401  (import triggers MPI init)

from harness import (  # noqa: E402
    STATE_NAMES, FORCING_NAMES,
    MetricsCollector, StubBackend,
    bootstrap_model, forward_step, read_state, zero_forcing,
)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-nml", "--namelist", required=True, help="Path to namelist file")
    p.add_argument("-dt", "--dt", type=float, default=300.0, help="Subcycle dt (s)")
    p.add_argument("--fm-x", type=float, default=1.0e-4,
                   help="Constant fm_x tendency requested from the (dummy) 'NN' (m/s^2)")
    p.add_argument("--quiet", action="store_true", help="Suppress model init output")
    return p.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)

    dims = bootstrap_model(args.namelist, quiet=args.quiet)
    print(f"dims: nelemd={dims.nelemd} np={dims.np_gll} nlev={dims.nlev}")

    # Initial state, on the CPU, float64. requires_grad so autograd tracks it as an input.
    state = read_state(dims, tl=0)
    for k in state:
        state[k].requires_grad_(True)

    # Forcing: a constant fm_x, everything else zero. requires_grad so we can inspect
    # d loss / d forcing after backward().
    forcing = zero_forcing(dims)
    forcing["fm_x"] = torch.full_like(forcing["fm_x"], args.fm_x)
    for k in forcing:
        forcing[k].requires_grad_(True)

    backend = StubBackend()
    metrics = MetricsCollector()

    state_np1 = forward_step(state, forcing, args.dt, backend, dims, metrics, step_idx=0)

    # Trivial scalar loss: sum of the u field at np1. Real training uses a proper loss;
    # this just gives autograd something to reduce so backward() runs.
    loss = state_np1["u"].sum()
    print(f"loss (sum of u_np1) = {float(loss):.6e}")

    loss.backward()

    # With StubBackend all input grads are zero. Check the plumbing: every input tensor
    # should have a .grad attached with the right shape.
    print("\ninput gradients (should all be zeros for StubBackend):")
    for name in STATE_NAMES:
        g = state[name].grad
        assert g is not None, f"state[{name}] has no grad — autograd plumbing broken"
        assert g.shape == state[name].shape, f"state[{name}] grad shape mismatch"
        assert torch.all(g == 0), f"StubBackend produced nonzero grad for state[{name}]"
        print(f"  d loss / d state[{name}]: shape={tuple(g.shape)} max|.|={float(g.abs().max()):.1e}")
    for name in FORCING_NAMES:
        g = forcing[name].grad
        assert g is not None, f"forcing[{name}] has no grad — autograd plumbing broken"
        assert g.shape == forcing[name].shape, f"forcing[{name}] grad shape mismatch"
        assert torch.all(g == 0), f"StubBackend produced nonzero grad for forcing[{name}]"
        print(f"  d loss / d forcing[{name}]: shape={tuple(g.shape)} max|.|={float(g.abs().max()):.1e}")

    print("\nmetrics: " + metrics.summary())
    print("\nOK")


if __name__ == "__main__":
    main(sys.argv)
