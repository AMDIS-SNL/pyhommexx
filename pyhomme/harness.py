"""P1 + P2: torch.autograd.Function around pyhommexx.forward(dt) + per-step instrumentation.

This is the first cut of the BPTT harness. It nails the interface shape so backend swaps
(stub -> FD -> dpfad -> real JtV) and metric additions land as drop-ins.

Design decisions made without further review — flag and change if any are wrong:
- State prognostics carried: u, v, w, vtheta_dp, phi, dp (6 fields per plan §5.1). dp is
  read/written but unforced (plan §3: no fdp exists in ElementsForcingST).
- Forcing carried: fm_x, fm_y, fm_z, fvtheta, fphi (matches plan §3, plan B1 scope). Tracer
  forcing (fq) deferred to B9/B11.
- Time-level convention: caller always reads/writes at tl=0. forward(dt) internally rotates
  levels via LEAPFROG; from Python's view, tl=0 is "current state" before and after.
- State is written to n0 only. nm1 is not touched. For a first step from a freshly-initialized
  model this is fine; multi-step training against interior states may need to seed nm1 too —
  revisit when we hit that in M3/M4.
- Backend interface (Protocol): backend.jtv(state_n0, forcing, dt, cotangent) returns
  (grad_state_n0, grad_forcing). StubBackend returns zeros. Real backends slot in without
  touching the autograd wrapper.
- Metrics scope: wall time per step + peak RSS (rusage) + bytes moved across the
  Python<->Kokkos boundary (feeds B8 measurement question). Checkpoint count is scaffolded
  but always 0 until Revolve (P5) lands.

Not in scope yet:
- Real backward (P4 FD, P5 checkpointing, B3 JtV).
- Tracer forcing (B9).
- Multi-rank memory accounting (needs MPI-aware collective; single-rank is fine for P3 sweeps
  at ne30 up to what fits on one node).
"""

from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import torch

sys.path.append(os.environ["PYHOMMEXX_LIB_PATH"])
import pyhommexx  # noqa: E402


# --- Field registry ---------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    name: str          # matches pyhommexx.{get,set}_state_var / {get,set}_forcing name
    kind: str          # "state" or "forcing"
    interface: bool    # True for interface-level fields (nlev+1), False for midpoints (nlev)


STATE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("u",         "state", False),
    FieldSpec("v",         "state", False),
    FieldSpec("w",         "state", True),
    FieldSpec("vtheta_dp", "state", False),
    FieldSpec("phi",       "state", True),
    FieldSpec("dp",        "state", False),
)

FORCING_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("fm_x",    "forcing", False),
    FieldSpec("fm_y",    "forcing", False),
    FieldSpec("fm_z",    "forcing", False),
    FieldSpec("fvtheta", "forcing", False),
    FieldSpec("fphi",    "forcing", True),
)

STATE_NAMES = tuple(f.name for f in STATE_FIELDS)
FORCING_NAMES = tuple(f.name for f in FORCING_FIELDS)


def _field_shape(spec: FieldSpec, nelemd: int, np_gll: int, nlev: int) -> tuple[int, int, int, int]:
    n = nlev + 1 if spec.interface else nlev
    return (nelemd, np_gll, np_gll, n)


# --- Model bootstrap --------------------------------------------------------

@dataclass
class ModelDims:
    nelemd: int
    np_gll: int
    nlev: int


def bootstrap_model(namelist: str, quiet: bool = False) -> ModelDims:
    """Init pyhommexx + read namelist + model_init + dp3d-from-ps. Idempotent per process."""
    pyhommexx.init_session(do_print_to_screen=not quiet)
    pyhommexx.enable_scalar_type("real")
    pyhommexx.read_params(namelist)
    pyhommexx.model_init()
    pyhommexx.init_dp3d_from_ps()

    params = pyhommexx.get_params()
    return ModelDims(
        nelemd=pyhommexx.get_nelemd(),
        np_gll=params["np"],
        nlev=params["nlev"],
    )


# --- Read / write helpers ---------------------------------------------------

def read_state(dims: ModelDims, tl: int = 0) -> dict[str, torch.Tensor]:
    """Copy current state out of pyhommexx into torch tensors (CPU float64)."""
    out: dict[str, torch.Tensor] = {}
    for spec in STATE_FIELDS:
        shape = _field_shape(spec, dims.nelemd, dims.np_gll, dims.nlev)
        arr = np.zeros(shape, dtype=np.float64)
        pyhommexx.get_state_var(arr, spec.name, "real", tl)
        out[spec.name] = torch.from_numpy(arr)
    return out


def write_state(state: dict[str, torch.Tensor], tl: int = 0) -> int:
    """Copy state tensors into pyhommexx. Returns total bytes written (for B8/P2)."""
    total = 0
    for spec in STATE_FIELDS:
        t = state[spec.name].detach().contiguous()
        arr = t.numpy()
        pyhommexx.set_state_var(arr, spec.name, "real", tl)
        total += arr.nbytes
    return total


def write_forcing(forcing: dict[str, torch.Tensor]) -> int:
    """Copy forcing tensors into pyhommexx. Returns total bytes written (for B8/P2)."""
    total = 0
    for spec in FORCING_FIELDS:
        t = forcing[spec.name].detach().contiguous()
        arr = t.numpy()
        pyhommexx.set_forcing(arr, spec.name, "real")
        total += arr.nbytes
    return total


def zero_forcing(dims: ModelDims) -> dict[str, torch.Tensor]:
    """A zero-valued forcing pytree of the right shapes. Convenient for tests."""
    return {
        spec.name: torch.zeros(_field_shape(spec, dims.nelemd, dims.np_gll, dims.nlev),
                               dtype=torch.float64)
        for spec in FORCING_FIELDS
    }


# --- Backward strategies ----------------------------------------------------

class Backend(Protocol):
    """A backend computes JtV: given upstream cotangent, produce input gradients.

    Called by _PyhommeForwardFn.backward. Must return two dicts with the same keys as
    STATE_NAMES and FORCING_NAMES respectively, tensors matching the corresponding
    input shapes.
    """
    name: str

    def jtv(
        self,
        state_n0: dict[str, torch.Tensor],
        forcing: dict[str, torch.Tensor],
        dt: float,
        cotangent: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        ...


@dataclass
class StubBackend:
    """Returns zero gradients. Exercises the autograd plumbing without any real math.

    Use for interface verification only. Any loss.backward() through this backend
    produces d loss / d anything == 0, which will NOT train an NN — swap for a real
    backend (P4 FD, then B3 JtV) once the plumbing is confirmed end-to-end.
    """
    name: str = "stub"

    def jtv(self, state_n0, forcing, dt, cotangent):
        gs = {k: torch.zeros_like(v) for k, v in state_n0.items()}
        gf = {k: torch.zeros_like(v) for k, v in forcing.items()}
        return gs, gf


# --- Metrics ----------------------------------------------------------------

@dataclass
class StepMetrics:
    step_idx: int
    wall_time_s: float
    peak_rss_mb: float
    state_bytes_in: int      # write_state pre-forward
    forcing_bytes_in: int    # write_forcing pre-forward
    state_bytes_out: int     # read_state post-forward
    checkpoint_count: int = 0  # scaffold; always 0 until P5 Revolve


@dataclass
class MetricsCollector:
    steps: list[StepMetrics] = field(default_factory=list)

    def record(self, m: StepMetrics) -> None:
        self.steps.append(m)

    def summary(self) -> str:
        if not self.steps:
            return "(no steps recorded)"
        wall = [s.wall_time_s for s in self.steps]
        peak = max(s.peak_rss_mb for s in self.steps)
        s_in = sum(s.state_bytes_in for s in self.steps)
        f_in = sum(s.forcing_bytes_in for s in self.steps)
        s_out = sum(s.state_bytes_out for s in self.steps)
        return (
            f"steps={len(self.steps)}  "
            f"wall(s) mean={np.mean(wall):.3f} max={max(wall):.3f} total={sum(wall):.3f}  "
            f"peak_rss={peak:.1f} MB  "
            f"bytes state_in={s_in:,} forcing_in={f_in:,} state_out={s_out:,}"
        )


def _peak_rss_mb() -> float:
    """Peak resident set size in MB. Linux ru_maxrss is KB; macOS bytes. Normalize to MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


# --- autograd nucleus -------------------------------------------------------

class _PyhommeForwardFn(torch.autograd.Function):
    """Positional-tensor autograd wrapper around pyhommexx.forward(dt).

    Positional inputs (all torch.Tensor, float64, on CPU):
        state:   u, v, w, vtheta_dp, phi, dp  (STATE_NAMES order)
        forcing: fm_x, fm_y, fm_z, fvtheta, fphi (FORCING_NAMES order)

    Non-tensor config goes via ctx (dt, backend, dims, metrics).

    Returns state at np1 as a tuple in STATE_NAMES order.
    """

    @staticmethod
    def forward(ctx, dt, backend, dims, metrics, step_idx, *tensors):
        n_state = len(STATE_NAMES)
        state_in = dict(zip(STATE_NAMES, tensors[:n_state]))
        forcing_in = dict(zip(FORCING_NAMES, tensors[n_state:]))

        ctx.dt = dt
        ctx.backend = backend
        ctx.dims = dims
        ctx.save_for_backward(*tensors)

        t0 = time.perf_counter()

        # Push state and forcing across the Python->Kokkos boundary.
        bytes_state_in = write_state(state_in, tl=0)
        bytes_forcing_in = write_forcing(forcing_in)

        # One subcycle. Internally: apply_cam_forcing -> RK stages -> vertical remap ->
        # LEAPFROG rotation. Afterwards, tl=0 points to what was np1.
        pyhommexx.forward(dt)

        # Pull the new state back.
        state_out = read_state(dims, tl=0)
        bytes_state_out = sum(v.numpy().nbytes for v in state_out.values())

        wall = time.perf_counter() - t0

        if metrics is not None:
            metrics.record(StepMetrics(
                step_idx=step_idx,
                wall_time_s=wall,
                peak_rss_mb=_peak_rss_mb(),
                state_bytes_in=bytes_state_in,
                forcing_bytes_in=bytes_forcing_in,
                state_bytes_out=bytes_state_out,
            ))

        return tuple(state_out[name] for name in STATE_NAMES)

    @staticmethod
    def backward(ctx, *grad_outs):
        tensors = ctx.saved_tensors
        n_state = len(STATE_NAMES)
        state_n0 = dict(zip(STATE_NAMES, tensors[:n_state]))
        forcing = dict(zip(FORCING_NAMES, tensors[n_state:]))
        cotangent = dict(zip(STATE_NAMES, grad_outs))

        grad_state, grad_forcing = ctx.backend.jtv(state_n0, forcing, ctx.dt, cotangent)

        # Return grads in the same positional order as forward's *inputs.
        # Non-tensor inputs (dt, backend, dims, metrics, step_idx) get None.
        return (
            None, None, None, None, None,
            *[grad_state[name] for name in STATE_NAMES],
            *[grad_forcing[name] for name in FORCING_NAMES],
        )


# --- ergonomic wrapper ------------------------------------------------------

def forward_step(
    state: dict[str, torch.Tensor],
    forcing: dict[str, torch.Tensor],
    dt: float,
    backend: Backend,
    dims: ModelDims,
    metrics: MetricsCollector | None = None,
    step_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """One differentiable subcycle. state and forcing are dicts keyed by field name.

    Returns a fresh state dict (post-forward). Gradients flow through both state and
    forcing inputs via backend.jtv.
    """
    _validate_pytree(state, STATE_NAMES, dims, "state")
    _validate_pytree(forcing, FORCING_NAMES, dims, "forcing")

    tensors = (
        *[state[name] for name in STATE_NAMES],
        *[forcing[name] for name in FORCING_NAMES],
    )
    out = _PyhommeForwardFn.apply(dt, backend, dims, metrics, step_idx, *tensors)
    return dict(zip(STATE_NAMES, out))


def _validate_pytree(d: dict[str, torch.Tensor], expected: tuple[str, ...],
                     dims: ModelDims, kind: str) -> None:
    missing = set(expected) - set(d.keys())
    extra = set(d.keys()) - set(expected)
    if missing or extra:
        raise ValueError(f"{kind} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    spec_by_name = {s.name: s for s in (STATE_FIELDS if kind == "state" else FORCING_FIELDS)}
    for name, t in d.items():
        expected_shape = _field_shape(spec_by_name[name], dims.nelemd, dims.np_gll, dims.nlev)
        if tuple(t.shape) != expected_shape:
            raise ValueError(f"{kind}[{name}] shape {tuple(t.shape)} != expected {expected_shape}")
        if t.dtype != torch.float64:
            raise ValueError(f"{kind}[{name}] dtype {t.dtype} != torch.float64")
