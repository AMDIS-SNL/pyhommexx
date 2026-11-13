#!/usr/bin/env python3
"""
Test of the adjoint of prim_step: forward sensitivities (dpfad) vs adjoint (prim_step_adj).

For one call to prim_step(dt), seen as a map x_N = F(x_0) of the dynamics state (u, v, vtheta_dp,
dp, w, phi) and tracers (qdp), we check the dot-product identity

    < dF * du0 , lamN >  ==  < du0 , dF^T * lamN >

where
 - dF * du0 is the forward tangent: prim_step run with the 'dpfad' scalar type, whose initial
   state carries the derivative along the (random) direction du0,
 - dF^T * lamN is the adjoint: prim_step_adj, from a checkpoint of the initial state, applied to
   a (random) seed lamN. Since the identity is linear in lamN, we check it separately for each
   field seeded alone: if something goes wrong, this tells us which field.
It is the python version of prim_step_adjoint_ut (in E3SM's homme unit tests), going through
the same pyhommexx entry points that the training loop will use. A scalar objective J of the final
state is just the special case lamN = dJ/dx_N: dJ/dp = <lamN, dF * du0> = <dF^T * lamN, du0>.

The model is configured via namelist_adjoint.nl, which differs from namelist.nl by the settings
the adjoint supports: qsplit=2 (2 dynamics steps per prim_step, to exercise their chaining), and
nu_top=0 (nu_p>0 is not supported either, but homme's init requires it for a namelist: the script
overrides it to 0 with set_params, between read_params and model_init). It also requires
time_step_type=10, limiter_option=9, hypervis_scaling=0, qsize>0 and Eulerian transport.

Requires PYHOMMEXX_LIB_PATH to point to the build dir that contains pyhommexx.*.so, and
a build with HOMMEXX_ENABLE_FAD_TYPES=ON (and without HOMMEXX_ENABLE_FWD_SENS).
Run it from this folder (the namelist uses relative paths for the vertical coordinate files).
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.environ["PYHOMMEXX_LIB_PATH"])
import pyhommexx  # noqa: E402

from mpi4py import MPI  # noqa: E402,F401  (import triggers MPI init)


# The dynamics prognostics, as named by pyhommexx. w and phi live at the interfaces.
FIELDS = ("u", "v", "vtheta_dp", "dp", "w", "phi")
INTERFACE_FIELDS = ("w", "phi")

# Number of derivatives of the dpfad type (HOMMEXX_DP_SFAD_SIZE, 1 by default). We only use the first
NDER = 1


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-nml", "--namelist", required=True, help="Path to namelist file (e.g., namelist_adjoint.nl)")
    p.add_argument("-dt", "--dt", type=float, default=300.0, help="Dynamics time step dt, in seconds")
    p.add_argument("--seed", type=int, default=1234, help="Seed of the random numbers generator")
    p.add_argument("--rtol", type=float, default=1.0e-6,
                   help="Max relative difference between the two sides of the identity")
    p.add_argument("--variant", choices=["both", "dyn", "q"], default="both",
                   help="Which initial perturbation the forward run carries")
    p.add_argument("--adjoint", action="store_true", help="Also run the adjoint (qdp seed only)")
    p.add_argument("--only", choices=list(FIELDS), default=None,
                   help="Perturb only this dynamics field (with variant dyn/both)")
    p.add_argument("--fd", type=float, nargs="*", default=[],
                   help="Also compute <dqN,lamN> by central finite differences of the Real prim_step, for these eps")
    p.add_argument("--qdp-init", choices=["smooth", "random"], default="smooth")
    return p.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    rng_q = np.random.default_rng(args.seed + 1)
    print(f"seed: {args.seed}", flush=True)

    # Ensure both real and dpfad are enabled: model_init then creates all the data structures
    # and functors for both scalar types
    pyhommexx.init_session(do_print_to_screen=False)
    pyhommexx.enable_scalar_type("real")
    pyhommexx.enable_scalar_type("dpfad")
    pyhommexx.read_params(args.namelist)
    # alloc_sphere_coords: needed for the lat/lon used below.
    # nu_p=0: the adjoint does not support nu_p>0 yet (R6), but homme's init rejects a namelist with nu_p=0
    pyhommexx.set_params({"alloc_sphere_coords": True, "nu_p": 0.0})
    pyhommexx.model_init()
    pyhommexx.init_dp3d_from_ps()

    params = pyhommexx.get_params()
    nelem = pyhommexx.get_nelemd()
    ngp = params["np"]
    nlev = params["nlev"]
    qsize = params["qsize"]
    print(f"nelemd={nelem} ne={params['ne']} np={ngp} nlev={nlev} qsize={qsize} "
          f"qsplit(dt_tracer_factor)={params['qsplit']}", flush=True)
    assert qsize > 0, "this test needs tracers (qsize>0)"

    def shape(name):
        return (nelem, ngp, ngp, nlev + 1 if name in INTERFACE_FIELDS else nlev)
    qshape = (nelem, qsize, ngp, ngp, nlev)

    # Turn on the adjoint (taping, adjoint state, and the tracers' adjoint state and checkpoint)
    pyhommexx.enable_adjoint()

    # ===== Initial state: the Real one is what model_init gave us, plus smooth positive tracers ===== #
    x0 = {}
    for name in FIELDS:
        x0[name] = np.zeros(shape(name))
        pyhommexx.get_state_var(x0[name], name, "real", 0)

    lat = np.zeros((nelem, ngp, ngp))
    lon = np.zeros((nelem, ngp, ngp))
    pyhommexx.get_dyn_latlon(lat, lon)
    qdp0 = np.zeros(qshape)
    for iq in range(qsize):
        # Smooth (so continuous across elements) and positive, so that the limiter stays away from
        # its zero floor and from its mass-relaxation branch (where the adjoint is approximate)
        q = 1.0e-3 * (1.0 + 0.5 * np.cos(lat) * np.sin((iq + 1) * lon + 0.3 * iq))
        if args.qdp_init == "random":
            q = 1.0e-3 * rng_q.uniform(0.5, 1.5, q.shape)
        qdp0[:, iq] = q[..., None] * x0["dp"]

    # The tracers' slots are not tied to n0/np1: prim_step reads qdp from slot_in, and writes to slot_out
    slot_in, slot_out = pyhommexx.get_prim_step_qdp_slots()
    pyhommexx.set_qdp(qdp0, "real", slot_in)

    # Checkpoint of the (Real) initial state, from which prim_step_adj will replay the fwd run
    pyhommexx.prim_step_checkpoint_take()

    # ===== Forward sensitivities: prim_step with the dpfad scalar type ===== #
    # Random tangent du0: for the dynamics it is x0*delta (perturb_state_var with factor=0 leaves
    # the value untouched, and sets the 1st derivative to x0*delta), for the tracers it's dq0.
    delta = {name: rng.uniform(-1.0, 1.0, shape(name)) for name in FIELDS}
    du0 = {name: x0[name] * delta[name] for name in FIELDS}
    dq0 = qdp0 * rng.uniform(-1.0, 1.0, qshape)
    if args.variant == "q":
        du0 = {name: 0.0 * du0[name] for name in FIELDS}
        delta = {name: 0.0 * delta[name] for name in FIELDS}
    if args.variant == "dyn":
        dq0 = 0.0 * dq0
    if args.only is not None:
        for name in FIELDS:
            if name != args.only:
                du0[name] = 0.0 * du0[name]
                delta[name] = 0.0 * delta[name]

    pyhommexx.copy_state("real", "dpfad")
    for name in FIELDS:
        pyhommexx.perturb_state_var(name, delta[name], 0.0, "dpfad")
    # (the bindings read the arrays as contiguous, so make sure they are)
    pyhommexx.set_qdp_dp_sens(np.ascontiguousarray(dq0[..., None]), slot_in)

    print("running prim_step (dpfad)...", flush=True)
    pyhommexx.prim_step(args.dt, "dpfad")

    duN = {}
    for name in FIELDS:
        arr = np.zeros(shape(name) + (NDER,))
        pyhommexx.get_state_var_dp_sens(arr, name)
        duN[name] = arr[..., 0].copy()
    arr = np.zeros(qshape + (NDER,))
    pyhommexx.get_qdp_dp_sens(arr, slot_out)
    duN["qdp"] = arr[..., 0].copy()

    # lamN is always drawn (same numbers in every variant)
    lamN = rng.uniform(-1.0, 1.0, qshape)
    rhs = np.vdot(duN["qdp"], lamN)
    # Also the pieces of the fwd result, to compare across runs
    print(f"[fwd {args.variant}] <dqN,lamN> = {rhs:.15e}", flush=True)
    print(f"[fwd {args.variant}] |dqN|_2 = {np.linalg.norm(duN['qdp']):.15e}", flush=True)
    # ===== Finite differences with the Real stack (independent check of the fwd Fad value) ===== #
    for eps in args.fd:
        qN = {}
        xN = {}
        for sign in (+1.0, -1.0):
            sin, sout = pyhommexx.get_prim_step_qdp_slots()
            for name in FIELDS:
                xp = x0[name] + sign * eps * du0[name]
                for tlv in (-1, 0, 1):
                    pyhommexx.set_state_var(np.ascontiguousarray(xp), name, "real", tlv)
            pyhommexx.set_qdp(np.ascontiguousarray(qdp0 + sign * eps * dq0), "real", sin)
            pyhommexx.prim_step(args.dt, "real")
            arr = np.zeros(qshape)
            pyhommexx.get_qdp(arr, "real", sout)
            qN[sign] = arr.copy()
            for name in FIELDS:
                a = np.zeros(shape(name))
                pyhommexx.get_state_var(a, name, "real", 1)  # np1, like get_state_var_dp_sens
                xN[(name, sign)] = a.copy()
        for name in FIELDS:
            dx_fd = (xN[(name, +1.0)] - xN[(name, -1.0)]) / (2.0 * eps)
            nf = np.linalg.norm(duN[name])
            if nf > 0:
                print(f"[fd {args.variant} eps={eps:.0e}] dyn out {name:>10}: |fwd fad|={nf:.6e} "
                      f"rel diff fd vs fad = {np.linalg.norm(dx_fd - duN[name]) / nf:.3e}", flush=True)
        dq_fd = (qN[+1.0] - qN[-1.0]) / (2.0 * eps)
        print(f"[fd {args.variant} eps={eps:.0e}] <dqN,lamN> = {np.vdot(dq_fd, lamN):.15e}   "
              f"|dqN|_2 = {np.linalg.norm(dq_fd):.15e}   "
              f"rel diff of dqN vs fwd fad: {np.linalg.norm(dq_fd - duN['qdp'])/np.linalg.norm(duN['qdp']):.3e}",
              flush=True)

    if not args.adjoint:
        pyhommexx.finalize()
        return

    print("running prim_step_adj (qdp seed)...", flush=True)
    pyhommexx.zero_adj_state()
    pyhommexx.zero_adj_qdp()
    pyhommexx.set_adj_qdp(lamN)
    pyhommexx.prim_step_adj(args.dt)
    lhs_dyn = 0.0
    for name in FIELDS:
        lam0 = np.zeros(shape(name))
        pyhommexx.get_adj_state_var(lam0, name)
        d = np.vdot(du0[name], lam0)
        lhs_dyn += d
        print(f"[adj] dyn part, field {name:>10}: <du0,lam0> = {d:.15e}", flush=True)
    lam0_q = np.zeros(qshape)
    pyhommexx.get_adj_qdp(lam0_q)
    lhs_q = np.vdot(dq0, lam0_q)
    print(f"[adj] qdp part: <dq0,lam0q> = {lhs_q:.15e}", flush=True)
    print(f"[adj] dyn part total       = {lhs_dyn:.15e}", flush=True)
    lhs = lhs_dyn + lhs_q
    print(f"[adj] total <du0,lam0>     = {lhs:.15e}", flush=True)
    pyhommexx.finalize()
    print(f"[cmp {args.variant}] adjoint {lhs:.15e}  fwd {rhs:.15e}  rel {abs(lhs-rhs)/max(abs(lhs),abs(rhs)):.3e}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
