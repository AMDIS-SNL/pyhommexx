#!/usr/bin/env python3
"""
Demonstrator for the B1/B2 forcing bindings.

What it does:
  1. Init the model from a namelist.
  2. Read u at n0 into u_before.
  3. Write a spatially-constant fm_x tendency into ElementsForcing via set_forcing_value.
  4. Call apply_dynamics_forcing(dt) once (bypasses the RK loop and the tracer path).
  5. Read u at n0 into u_after.
  6. Assert u_after - u_before == dt * fm_x, to roundoff.

Also exercises set_forcing (array write) as a second pass and re-verifies.

Requires PYHOMMEXX_LIB_PATH env var pointing at the build dir that contains pyhommexx.*.so.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.environ["PYHOMMEXX_LIB_PATH"])
import pyhommexx  # noqa: E402

from mpi4py import MPI  # noqa: E402,F401  (import triggers MPI init)


FM_X_CONST = 1.0e-4   # m/s^2
DT_DEFAULT = 300.0    # s
TOL = 1.0e-12


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-nml", "--namelist", required=True, help="Path to namelist file")
    p.add_argument("-dt", "--dt", type=float, default=DT_DEFAULT, help="Forcing dt in seconds")
    p.add_argument("--fm-x", type=float, default=FM_X_CONST, help="Constant fm_x tendency (m/s^2)")
    return p.parse_args(argv[1:])


def check_scalar_forcing(dt, fm_x, tag):
    """Assert u(n0)_after == u(n0)_before + dt*fm_x, and v is unchanged."""
    nelemd = pyhommexx.get_nelemd()
    params = pyhommexx.get_params()
    ngp = params["np"]
    nlev = params["nlev"]

    u_before = np.zeros((nelemd, ngp, ngp, nlev), dtype=np.float64)
    v_before = np.zeros_like(u_before)
    pyhommexx.get_state_var(u_before, "u", "real", 0)
    pyhommexx.get_state_var(v_before, "v", "real", 0)

    pyhommexx.apply_dynamics_forcing(dt, "real")

    u_after = np.zeros_like(u_before)
    v_after = np.zeros_like(u_before)
    pyhommexx.get_state_var(u_after, "u", "real", 0)
    pyhommexx.get_state_var(v_after, "v", "real", 0)

    du = u_after - u_before
    dv = v_after - v_before
    expected = dt * fm_x

    err_u = float(np.max(np.abs(du - expected)))
    err_v = float(np.max(np.abs(dv)))
    print(f"[{tag}] max|du - dt*fm_x| = {err_u:.3e}   max|dv| = {err_v:.3e}")

    # Scale-relative tolerance for u check (protects against |expected|=0 edge case),
    # absolute for v.
    assert err_u <= TOL * max(abs(expected), 1.0), \
        f"[{tag}] u forcing did not apply as expected: max err {err_u:.3e}"
    assert err_v <= TOL, f"[{tag}] v drifted despite fm_y=0: max err {err_v:.3e}"


def main(argv):
    args = parse_args(argv)

    pyhommexx.init_session(do_print_to_screen=True)
    pyhommexx.enable_scalar_type("real")
    pyhommexx.read_params(args.namelist)
    pyhommexx.model_init()
    pyhommexx.init_dp3d_from_ps()

    nelemd = pyhommexx.get_nelemd()
    params = pyhommexx.get_params()
    print(f"nelemd={nelemd} ne={params['ne']} np={params['np']} nlev={params['nlev']}")

    # Pass 1: use set_forcing_value to write a spatially-constant fm_x.
    # Also explicitly zero fm_y, fm_z, fvtheta, fphi so we can assert v is unchanged.
    for name in ("fm_x", "fm_y", "fm_z", "fvtheta", "fphi"):
        pyhommexx.set_forcing_value(0.0, name, "real")
    pyhommexx.set_forcing_value(args.fm_x, "fm_x", "real")
    check_scalar_forcing(args.dt, args.fm_x, tag="set_forcing_value")

    # Pass 2: same test, but write the forcing as an ndarray via set_forcing.
    # ElementsForcing fields have no time-level dimension.
    ngp = params["np"]
    nlev = params["nlev"]
    fm_x_arr = np.full((nelemd, ngp, ngp, nlev), args.fm_x, dtype=np.float64)
    zeros_mid = np.zeros((nelemd, ngp, ngp, nlev), dtype=np.float64)
    zeros_int = np.zeros((nelemd, ngp, ngp, nlev + 1), dtype=np.float64)
    pyhommexx.set_forcing(fm_x_arr, "fm_x", "real")
    pyhommexx.set_forcing(zeros_mid, "fm_y", "real")
    pyhommexx.set_forcing(zeros_mid, "fm_z", "real")
    pyhommexx.set_forcing(zeros_mid, "fvtheta", "real")
    pyhommexx.set_forcing(zeros_int, "fphi", "real")

    # Round-trip: read fm_x back and confirm it matches what we wrote.
    fm_x_readback = np.zeros_like(fm_x_arr)
    pyhommexx.get_forcing(fm_x_readback, "fm_x", "real")
    rt_err = float(np.max(np.abs(fm_x_readback - fm_x_arr)))
    print(f"[set_forcing/get_forcing] round-trip max err = {rt_err:.3e}")
    assert rt_err <= TOL * max(abs(args.fm_x), 1.0), \
        f"forcing round-trip failed: max err {rt_err:.3e}"

    check_scalar_forcing(args.dt, args.fm_x, tag="set_forcing")

    print("OK")
    pyhommexx.finalize()


if __name__ == "__main__":
    main(sys.argv)
