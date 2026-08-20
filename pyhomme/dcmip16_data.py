"""Data loader for DCMIP-2016 test 1 hifreq training set (E9/E10).

Reads NetCDF files produced by
`components/homme/dcmip_tests/dcmip2016_test1_baroclinic_wave/theta-l/`
with `namelist-dcmip16-ne{2,30}-hifreq.nl`, yielding (state, tendency) pairs
for D2 offline training and (state_k, state_k+N) trajectories for F3.

Ground-truth tendencies from the physics wrapper are `FM_x`, `FM_y`, `FM_z`
(momentum, m/s^2), `FT` (temperature, K/s), and `FQ1..FQ3` (tracer mass
tendency for qv, qc, qr in units of kg/(m^2 s), already dp-weighted per
`dcmip16_wrapper.F90:576`).

pyhommexx's forcing plumbing (harness.STATE_FIELDS / FORCING_FIELDS, B1 scope)
uses `fm_{x,y,z}`, `fvtheta = dp * d(theta_v)/dt`, and `fphi`. Two conversion
gaps this module handles:

  1. `FT` (dT/dt) -> `fvtheta`. Approximation used here: hold pressure fixed
     across the physics step (true to O(dt) for se_ftype=0 where dynamics runs
     after forcing), so `dtheta/dt ~= (1/exner) * dT/dt`, then
     `fvtheta ~= dp * (1 + Mvap*qv) * dtheta/dt`.
     Exner is derived from output `pnh` and `dp`. See `_ft_to_fvtheta`.

  2. `FQ` heads have no pyhommexx binding yet (B9 blocked on B11 tracer
     registry). We still emit the target tensors so D8 evaluation can start
     when the bindings land; the current offline D2 loop just ignores them
     unless they appear in the active head set.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


# All heads the NN could produce, in canonical order.
ALL_HEADS: tuple[str, ...] = (
    "fm_x", "fm_y", "fm_z", "fvtheta",
    "fq_qv", "fq_qc", "fq_qr",
)
# Heads currently bindable through pyhommexx forcing. Extend when B9 lands.
BINDABLE_HEADS: tuple[str, ...] = ("fm_x", "fm_y", "fm_z", "fvtheta")

# State fields the NN reads. Matches harness.STATE_FIELDS layout except we
# also pull temperature-adjacent diagnostics we need for FT->fvtheta conversion.
INPUT_STATE_VARS: tuple[str, ...] = ("u", "v", "w", "Th", "geo", "dp")


# Physical constants matched to E3SM's physical_constants module.
_RGAS = 287.04
_CP = 1004.64
_KAPPA = _RGAS / _CP
_P0 = 100000.0
_RWATER_VAPOR = 461.5
_MVAP = _RWATER_VAPOR / _RGAS - 1.0  # ~0.608


def _ft_to_fvtheta(ft: np.ndarray, pnh: np.ndarray, dp: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """Convert FT (K/s) to fvtheta (dp*dK/dt on theta_v). See module docstring."""
    exner = (pnh / _P0) ** _KAPPA
    dtheta_dt = ft / exner
    return dp * (1.0 + _MVAP * qv) * dtheta_dt


@dataclass
class Sample:
    """One (state, tendency) pair on native GLL grid, flattened over (elem, np, np) -> ncol."""
    state: dict[str, torch.Tensor]      # (ncol, nlev_or_nlevp) per field
    tendency: dict[str, torch.Tensor]   # (ncol, nlev) per head in ALL_HEADS
    time_index: int


class TendencyDataset(Dataset):
    """(state_k, tendency_k) pairs from one or more hifreq NetCDF files.

    Args:
        paths: glob pattern(s) matching NetCDF file(s) from E9.
        head_names: which heads to expose in `sample.tendency`. Default is
            BINDABLE_HEADS; pass ALL_HEADS to also get FQ targets (data-only
            until B9 binds them). Unknown names raise.
        stride: subsample every `stride`-th time index (1 = every snapshot).
    """

    def __init__(
        self,
        paths: str | Sequence[str],
        head_names: Sequence[str] = BINDABLE_HEADS,
        stride: int = 1,
    ):
        if isinstance(paths, str):
            paths = sorted(glob.glob(paths))
        if not paths:
            raise FileNotFoundError(f"no files matched: {paths}")
        for h in head_names:
            if h not in ALL_HEADS:
                raise ValueError(f"unknown head {h!r}; valid: {ALL_HEADS}")
        self._head_names = tuple(head_names)

        # xarray opens lazily and dask'd — fine for large hifreq output.
        self._ds = xr.open_mfdataset(list(paths), combine="by_coords")
        self._time_ix = np.arange(0, self._ds.sizes["time"], stride)

    def __len__(self) -> int:
        return len(self._time_ix)

    def __getitem__(self, i: int) -> Sample:
        t = int(self._time_ix[i])
        snap = self._ds.isel(time=t)

        state = {v: _flatten_columns(snap[v].values) for v in INPUT_STATE_VARS}

        # tendency conversion
        ncol, nlev = state["u"].shape
        tend_np: dict[str, np.ndarray] = {}
        for head in self._head_names:
            if head == "fm_x":
                tend_np[head] = _flatten_columns(snap["FM_x"].values)
            elif head == "fm_y":
                tend_np[head] = _flatten_columns(snap["FM_y"].values)
            elif head == "fm_z":
                tend_np[head] = _flatten_columns(snap["FM_z"].values)
            elif head == "fvtheta":
                ft  = _flatten_columns(snap["FT"].values)
                pnh = _flatten_columns(snap["pnh"].values)
                dp  = _flatten_columns(snap["dp"].values)
                qv  = _flatten_columns(snap["Q"].values)
                tend_np[head] = _ft_to_fvtheta(ft, pnh, dp, qv)
            elif head == "fq_qv":
                tend_np[head] = _flatten_columns(snap["FQ1"].values)
            elif head == "fq_qc":
                tend_np[head] = _flatten_columns(snap["FQ2"].values)
            elif head == "fq_qr":
                tend_np[head] = _flatten_columns(snap["FQ3"].values)

        return Sample(
            state={k: torch.from_numpy(v).float() for k, v in state.items()},
            tendency={k: torch.from_numpy(v).float() for k, v in tend_np.items()},
            time_index=t,
        )


def _flatten_columns(a: np.ndarray) -> np.ndarray:
    """(...,nlev) native-GLL field -> (ncol, nlev). Handles both interp'd and native layouts."""
    if a.ndim == 3:                                # (nelemd*np*np, nlev) already flat, or (lat, lon, nlev)
        return a.reshape(-1, a.shape[-1])
    if a.ndim == 4:                                # (nelemd, np, np, nlev)
        nelemd, npi, npj, nlev = a.shape
        return a.reshape(nelemd * npi * npj, nlev)
    if a.ndim == 2:                                # already (ncol, nlev)
        return a
    raise ValueError(f"unexpected field shape {a.shape}")
