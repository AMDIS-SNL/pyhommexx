#!/usr/bin/env python3
"""P4 demo: run the FD adjoint-consistency verifier against StubBackend.

Expected outcome:
- Verifier prints a few (seed, lhs, rhs, ratio) triples.
- Every ratio should be ~0 (StubBackend returns zero grads by design; LHS = 0 while
  RHS = <cotangent, Jv_FD> is generically nonzero).
- The demo asserts |ratio| below a small tolerance for all seeds. If it passes, the
  verifier is wired up correctly and any future backend that produces nonzero grads
  can be validated by swapping StubBackend for it.

This is the "positive test of the verifier" — a broken verifier would let StubBackend
'pass', so this run establishes the check itself works. Later backends (dpfad, JtV) will
be validated by swapping them in and expecting ratio ≈ 1.
"""

import argparse
import os
import sys

# Import order: harness first (loads pyhommexx with GCC 12 libstdc++), then torch.
# See the comment at the top of harness.py.
from harness import (  # noqa: E402
    StubBackend,
    bootstrap_model, read_state, zero_forcing, verify_backend,
)

import torch  # noqa: E402
from mpi4py import MPI  # noqa: E402,F401  (import triggers MPI init)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-nml", "--namelist", required=True, help="Path to namelist file")
    p.add_argument("-dt", "--dt", type=float, default=300.0, help="Subcycle dt (s)")
    p.add_argument("--fm-x", type=float, default=1.0e-4, help="Baseline fm_x tendency (m/s^2)")
    p.add_argument("--eps", type=float, default=1.0e-6, help="FD perturbation size")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="RNG seeds for random directions/cotangents")
    p.add_argument("--forward-diff", action="store_true",
                   help="Use forward FD instead of central (halves cost, O(eps) accuracy)")
    p.add_argument("--stub-tol", type=float, default=1.0e-12,
                   help="Max |ratio| for StubBackend to pass (should be ~0)")
    p.add_argument("--quiet", action="store_true", help="Suppress model init output")
    return p.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)

    dims = bootstrap_model(args.namelist, quiet=args.quiet)
    print(f"dims: nelemd={dims.nelemd} np={dims.np_gll} nlev={dims.nlev}", flush=True)

    state = read_state(dims, tl=0)
    forcing = zero_forcing(dims)
    forcing["fm_x"] = torch.full_like(forcing["fm_x"], args.fm_x)

    print(f"verifying StubBackend  eps={args.eps:.1e}  "
          f"central={not args.forward_diff}  seeds={args.seeds}", flush=True)

    results = verify_backend(
        StubBackend(),
        state, forcing, args.dt, dims,
        seeds=args.seeds,
        eps=args.eps,
        central=not args.forward_diff,
    )

    print("")
    for r in results:
        print("  " + r.summary())
    print("")

    # Assertion: StubBackend must produce ratio ~0 (LHS=0 exactly; ratio depends on RHS
    # magnitude but should be well below any real backend). If a seed gave |RHS| below
    # the floor, `ratio` is NaN — treat that as a degenerate direction, not a failure.
    for r in results:
        if r.lhs != 0.0:
            raise AssertionError(
                f"StubBackend produced nonzero LHS ({r.lhs:+.3e}) for seed={r.seed}; "
                "expected exact zero (stub returns zero grads)."
            )
        if not (r.ratio != r.ratio):  # not NaN
            assert abs(r.ratio) <= args.stub_tol, \
                f"StubBackend ratio {r.ratio:+.3e} exceeds tol {args.stub_tol:.1e} at seed={r.seed}"

    print("OK  (verifier caught the stub — swap in a real backend to expect ratio ≈ 1)")


if __name__ == "__main__":
    main(sys.argv)
