# pyhommexx Development Plan

Hybrid forward PDE solver + ML for climate modeling: HommeXX dynamical core coupled to neural-network tendencies, trained from ERA5.

**Pinned references:** `AMDIS-SNL/E3SM` @ `920cf1e` (master), `E3SM-Project/HICCUP` @ 2026-08-14. Source citations below refer to `components/homme/` in the E3SM fork.

**Ownership:** this repo owns bindings, NN, data pipeline, training loop, BPTT harness, and CI. The AMDIS-SNL/E3SM fork owns adjoint kernels, `JtV` coverage, and FAD scalar types (§4).

---

## 1. Configuration

| Parameter | Value |
|---|---|
| Resolution | ne30 initially; ne120 intermediate; ne256 ultimately |
| Vertical levels | 128 (`PYHOMMEXX_NLEV`) |
| Hydrostatic mode | Non-hydrostatic |
| Tracers | `qsize` runtime-configurable; `QSIZE_D` ceiling compile-time |
| NN forcing | 5 of 6 dynamics prognostics + configurable tracers; `dp3d` unforced |
| GLL duplicates | DSS reconciles |
| `se_ftype` | 0 |
| Dev loop | ne2 / nlev128 — iteration only, not an extrapolation basis |

---

## 2. Design decisions

**DSS reconciliation.** With a column-local NN, duplicated GLL points carry identical column states, so tendencies match and DSS averaging is a no-op to roundoff. This holds only while the NN stays column-local; a horizontal receptive field makes DSS a real operator and its transpose a real gradient term. Recorded as an architectural constraint (D5), asserted in T3.

**`se_ftype=0`.** `prim_step.cpp:141` applies FORCING_0 unconditionally; line 164 calls `apply_cam_forcing_dynamics(dt_remap)`, line 148 `apply_cam_forcing_tracers(dt_q)`. Dynamics and tracer forcing therefore run on different timescales. If the NN emits both, its two output heads mean different things.

**qsize.** `PYHOMMEXX_QSIZE` sets `QSIZE_D`, the compile-time maximum; namelist `qsize` is the runtime actual. Raising the ceiling requires a rebuild, so set it generously (A7).

**Optimizer backend: PyTorch-native first, ROL deferred.** PyROL requires the `binder` Clang-libTooling code generator to pre-generate its pybind11 bindings — a second, Clang-specific toolchain running alongside this project's own nanobind bindings. First efforts (F3–F5) drive the differentiable `forward()` (P1) directly with `torch.optim` — Adam as the baseline, LBFGS as a quasi-Newton step up. This is not a decision to drop ROL permanently: its trust-region/Newton-CG optimizers and distributed vector support may still matter once BPTT crosses the single-node line at ne120+ (§5.3.4), which is exactly the regime M6 targets. F6 revisits PyROL if F3–F5 prove insufficient; A9 tracks the toolchain prerequisite for that path. Separately — and unconfirmed — plain `Trilinos_ENABLE_ROL` (C++ only, no PyROL/binder/pybind11) may still be worth enabling if the C++ team's adjoint work depends on it; flagged for follow-up.

---

## 3. Forcing coverage

`ForcingFunctor::states_forcing` applies, under NH:

```
u += dt*fm_x    v += dt*fm_y    w += dt*fm_z
vtheta += dt*fvtheta            phi += dt*fphi
```

then overwrites surface `w` from the `u`,`v` boundary condition.

**`dp3d` is unforced.** `ElementsForcingST` declares `m_fm`, `m_fvtheta`, `m_ft`, `m_fphi` — no `fdp`. Dry-air mass is conserved by construction. B1's shape is fixed and needs no new C++.

**`fvtheta` applies to `vtheta_dp`, not `vtheta`.** Because `dp3d` is unforced, `dp` is invariant under forcing and the conversion is exact:

```
fvtheta = dp · (dθv/dt)
```

The NN carries a θv-tendency head; the adapter multiplies by current `dp`. A missing `dp` factor yields a plausible model wrong by a level-dependent factor of ~10³.

**`m_ft` is allocated but unused by `states_forcing`.** Likely consumed in the tracer path. Do not bind it as a live NN output before confirming semantics (B10).

**Tracer forcing is runtime-configurable but moves `ps` and `dp3d`.** `Tracers.hpp:45` declares `fq`, applied by `tracers_forcing(dt, np1, np1_qdp, adjustment, use_moisture)` guarded on runtime `m_qsize`. The `TagTracersPre` kernel ("temperature, NH perturb press, FQps") and `TagTracersPost` update `m_dp3d` and `m_ps_v`. Consequently:

- Dry-air mass is safe — no `fdp` exists.
- Water mass is not. Forcing `qv` moves `ps` and `dp3d` through `FQps`. This is physically correct; real moist physics does the same. See open question 1.

**The tracer limiter is not differentiable.** `compute_fqdt` clips `fqdt` to `-qdp` (or zero) when `qdp + fq·dt < 0`. A hard kink with a zero-gradient region in the backward path, activating exactly where the NN drives tracers negative. Mitigation: penalize approach-to-negative in the loss (D4), so the NN is steered before the limiter fires.

---

## 4. C++ team dependencies

Merged today (`master`): CAAR `run_JV`, `run_JV_full`, `run_JtV`, `run_JtV_surf_bc`, `run_JtV_full` via Sacado `DxFadTypeCaar = SFad<double,16*NP*NP>`; `DxFadTypeDirk`; unit tests `caar_sacado_ut`, `eos_sacado_ut`, `dirk_sacado_ut`, `tridiag_sacado_ut`.

In progress, not merged: vertical remap, Eulerian transport, hyperviscosity, and RK-stage adjoint composition. Branch tips carry "BKP: this commit will be amended/erased/rewritten" — do not pin to those SHAs.

**`Tape.hpp`** (branch `bartgol/rk-adjoint-stepping`) is a preallocated fixed-capacity stack of `StateSnapshot` with `shift_fwd`/`shift_bwd`, driven from a new `prim_advance_exp.hpp`. Two implications:

1. It stores, it does not recompute — `tape.resize(capacity, ...)` allocates `capacity` full snapshots up front. Memory is `capacity × snapshot size` (§5.1). It bounds within-step differentiation depth; it cannot bound a 72-step window.
2. Revolve-style checkpointing therefore belongs above it, in our harness, treating a taped `forward()` as its atomic unit. The layers compose.

`Tape` uses `std::vector` (host memory), so on GPU builds each push/pop is a device↔host transfer.

**What we need from them, in order:**

1. `JtV` signature and `StateSnapshot` lifetime for the Python wrapper.
2. Confirmation that `ForcingFunctor` is as generic in `ST` as `ElementsForcingST` already is. The project rests on differentiating through the forcing application.
3. `Tape` capacity limits and how they surface to Python.
4. Confirmation the adjoint targets the ftype=0 path.
5. `JtV` memory behavior at NLEV=128 — `run_JV_full`/`run_JtV_full` require FAD size scaled to level count.
6. Adjoint treatment of the tracer limiter kink (§3).

Note: this work is needed regardless of optimizer backend (§2) — `JtV` produces the gradient BPTT consumes; PyTorch-native optimizers and ROL both need it.

---

## 5. BPTT scaling

Assumptions: double precision, GLL-redundant storage (`6·ne² × np² × nlev`, np=4), nlev=128, qsize=10, one time level per checkpoint, functor buffers excluded. These are lower bounds.

### 5.1 One state snapshot

| | ne2 | ne30 | ne120 | ne256 |
|---|---|---|---|---|
| Elements | 24 | 5,400 | 86,400 | 393,216 |
| Unique columns | ~218 | ~48,602 | ~777,602 | ~3,538,946 |
| Per 3D field | 0.4 MB | 88 MB | 1.4 GB | 6.4 GB |
| 6 dynamics prognostics | 2.4 MB | 531 MB | 8.5 GB | 39 GB |
| + 10 tracers | 6.3 MB | 1.4 GB | 22.6 GB | 103 GB |

### 5.2 A 6-hour window

Step counts assume dt scales inversely with grid spacing from the ne30 anchor of 300 s. Order-of-magnitude only; replace with measured dt where available.

| | ne2 | ne30 | ne120 | ne256 |
|---|---|---|---|---|
| Approx. dt | 300 s | 300 s | ~75 s | ~35 s |
| Steps per 6 h | 72 | 72 | 288 | 617 |
| Store every step | 0.5 GB | ~100 GB | ~6.5 TB | ~63 TB |
| 10-step window | 63 MB | ~14 GB | ~226 GB | ~1.0 TB |
| 2-step window | 13 MB | ~2.8 GB | ~45 GB | ~206 GB |

### 5.3 Consequences

1. **Store-every-step checkpointing is infeasible at ne30.** Binomial/Revolve checkpointing is the design, not a later optimization (Griewank & Walther, *ACM TOMS* 26(1):19–45, 2000, "Algorithm 799: revolve"). P5 is scoped as "implement Revolve."
2. **ne2 is not an extrapolation basis** — 225× smaller than ne30, 16,384× smaller than ne256. Fine for correctness and plumbing; misleading on memory and compute/communication balance. P3 sweeps at ne30.
3. **The 6-hour interval may not be the training window.** 288 steps at ne120 and 617 at ne256 are out of reach for full BPTT. Short-window training against interpolated targets, or multiple shooting, may be the realistic path. P6 answers this.
4. **ne120 is where BPTT stops being a single-node problem.** A snapshot with tracers is ~22.6 GB; a two-entry tape does not fit on one GPU. Cost compounds twice over ne30 — 16× elements and 4× steps — so a fixed-wall-clock window shrinks ~64×. Distributed checkpoint placement stops being deferrable here. This is also the regime where ROL's distributed vector support (§2) could matter for F4/F6.

---

## 6. Tasks

Sizes: S ≈ days, M ≈ 1–2 weeks, L ≈ 3+ weeks.

### A. Environment and build

| ID | Task | Size | Status |
|---|---|---|---|
| A1 | Build reproduced; `ctest -R sacado` green | S | Done (GitHub CI) |
| A2 | Fix `env/README.md`: venv created `amdis-venv`, activated `amdis-env`; `python -m install` → `python -m pip install` | S | Done |
| A3 | `Trilinos_ENABLE_ROL=ON` + PyROL in `configure-trilinos.sh`. Source build; no PyPI distribution | M | Deferred — see §2; not required for M0 |
| A4 | Pin CI to a known-good E3SM SHA instead of floating `master` | S | Won't do — we're the only consumer of this fork's `master`; pinning buys no protection |
| A5 | Promote CI smoke-import to a real `forward()` gate — `continue-on-error: true` today | S | To-do |
| A6 | Document CI-vs-cluster environment gap (CI: CPU/apt; cluster: modules, `/projects/amdis/tpl`, CUDA-capable) | S | Done |
| A7 | Choose and document the `QSIZE_D` ceiling; rebuild required to change it | S | Blocked — waiting on the tracer set from the data pipeline (E-series) |
| A8 | ne30 build + run configuration, distinct from the ne2 dev loop | M | To-do |
| A9 | If F6 triggers: resolve the PyROL build toolchain conflict — `binder` needs a matched Clang/LLVM, project targets gcc/Intel + nanobind. Confirm with C++ team whether plain `Trilinos_ENABLE_ROL` (no PyROL) is separately needed first | M | Deferred — contingent on F6 |

### B. Bindings

| ID | Task | Size |
|---|---|---|
| B1 | `set_forcing`/`get_forcing` for `fm`(3), `fvtheta`, `fphi` over `ElementsForcingST<ST>`. Critical path to every milestone | M |
| B2 | Bind `apply_cam_forcing_dynamics` / `states_forcing` for separate, testable application | S |
| B3 | Wrap `JtV` once its signature settles (§4) | S |
| B4 | Expose stage boundaries — `forward()` hides the whole `prim_run_subcycle`, leaving nowhere to checkpoint | M |
| B5 | Fix naming: `"vth"` (`pyhommexx_state.cpp:73,199`) vs `"vthetadp"` (`:431`) | S |
| B6 | Fix `theta-nlev128.py`: `u *= factoperturb`, undefined name, dead `--perturb` path | S |
| B7 | Replace hardcoded `PYHOMMEXX_LIB_PATH` with an env var | S |
| B8 | Zero-copy audit of `get_state_var`/`set_state_var` — at ne30 this moves ~1.4 GB per call if it copies | S |
| B9 | Configurable tracer forcing over `Tracers::fq`: runtime `qsize`, per-tracer enable mask, `[qsize, nlev]` shape, `dt_q` timescale | M |
| B10 | Resolve `m_ft` semantics before binding it | S |
| B11 | Tracer registry: dataset variable name ↔ Homme tracer index, in config and persisted with the model checkpoint. Homme tracers are bare indices with no names; a mismatch is silent | M |

### P. BPTT instrumentation

| ID | Task | Size |
|---|---|---|
| P1 | `torch.autograd.Function` around `forward(dt)` with pluggable backward (stub → FD → `dpfad` → real `JtV`) | M |
| P2 | Instrumentation: per-step wall time, peak memory, state-copy bytes, checkpoint count | M |
| P3 | Window sweep at ne30 (1, 2, 5, 10, 20 steps); ne2 for iteration only | M |
| P4 | FD ground truth at ne2 for verifying later backends | M |
| P5 | Across-step Revolve/binomial checkpointing, layered above the C++ `Tape`, treating a taped `forward()` as atomic | L |
| P6 | Pain-point report — the deliverable of this group | S |
| P7 | ne30 → ne120 → ne256 extrapolation; two measured rungs | S |
| P8 | Evaluate short-window / multiple-shooting alternatives to full 6h BPTT | M |
| P9 | Coordinate with C++ team on `Tape` capacity and Python exposure | S |

### D. Neural network

| ID | Task | Size |
|---|---|---|
| D1 | Column extraction `(nelemd,np,np,nlev)` ↔ `(ncol,nlev)`, DSS-consistent | M |
| D2 | Per-column MLP → `fm`(3), `fvtheta`, `fphi`; θv head scaled by `dp` (§3); document the dynamics-vs-tracer timescale split | M |
| D3 | Per-level, per-variable normalization, persisted with the checkpoint | S |
| D4 | Stability guardrails: clipping, NaN traps, fail-fast; loss term penalizing approach-to-negative tracers (§3) | M |
| D5 | Record column-locality as an architectural constraint (§2) | S |
| D7 | Configurable output heads driven by the active tracer set, not a fixed architecture | M |

### E. Data pipeline

| ID | Task | Size | Status |
|---|---|---|---|
| E1 | HICCUP env; `create_EAMxx_IC_from_ERA5-NOAA.py` at ne30 with `vert_coord_E3SM_L128.nc` | M | |
| E2 | Trajectories via `get_nudging_data.ERA5.py` (not the hindcast script) at 6h | M | |
| E3 | EAM → theta-l NH conversion: `T,PS,U,V,Q` → `vtheta_dp`, `dp3d`, balanced `phinh_i`, `w_i` init. Largest item we own | L | |
| E4 | Thin adapter importing `hiccup` — no fork; it ships `setup.py`, `remap_vertical_py`, and state-adjustment routines | M | |
| E5 | Round-trip test via `set_state_var`; short forward run measuring initialization shock | M | |
| E6 | CDS API tokens | S | Done |
| E7 | Cache converted trajectories; do not re-run HICCUP per epoch | M | |
| E8 | Confirm whether CI/automation needs a service-account CDS token | S | |

### F. Optimization

| ID | Task | Size | Status |
|---|---|---|---|
| F1 | ~~Spike: torch `state_dict` ↔ PyROL vector round-trip, before F2–F4 are scheduled~~ | M | Superseded — PyROL deferred, see §2 |
| F2 | ~~ROL objective wrapping coupled forward + backward~~ | L | Deferred — revisit as F6 |
| F3 | Serial training loop with `torch.optim.Adam`, driven directly off `forward()` (P1). Promoted from comparison baseline to primary path | M | |
| F4 | Distributed parameter-handling design note — also scopes whether ne120+ (§5.3.4) needs ROL's parallel vector support | M | |
| F5 | `torch.optim.LBFGS` (quasi-Newton, line-search) as a step up from Adam for M6, ahead of reintroducing ROL | M | New |
| F6 | Revisit PyROL if F3–F5 don't meet M6's bar — either 2nd-order convergence robustness or distributed vector support (§2, §5.3.4). Gates on A9 | M | Deferred |

### T. Testing and CI

| ID | Task | Size |
|---|---|---|
| T1 | Python test suite for new bindings | M |
| T2 | Nightly gradient regression once a real backend lands | S |
| T3 | Assert duplicated-point states agree to roundoff at extraction | S |
| T4 | ne2 regression: fixed nlev, few steps, bit-comparable | M |
| T5 | Document the ne2 dev loop and its extrapolation limits | S |

---

## 7. Milestones

| Milestone | Contents | Exit criterion |
|---|---|---|
| M0 — Foundations | A2, A5–A8, B10, B11, §4 + §9 questions sent (A3 deferred, not required) | `ft` clarified; tracer registry designed; `ps`-drift question answered; ne30 config builds |
| M1 — Tendencies injectable | B1, B2, B5–B8, D1, D5, T1, T3 | A constant tendency from Python provably changes the forward solution |
| M2 — Initialized from ERA5 | E1–E5, E7, E8, T4, T5 | ne30 runs N steps from a real ERA5 state without blowing up |
| M3 — BPTT harness + report | P1–P4, P6, P8, B4 | Report measured at ne30, with ne2 cross-checks |
| M4 — Offline-trained hybrid | D2–D4, D7, B9, F3 | NN trained on tendency matching; coupled run; drift measured vs. ERA5 |
| M5 — Real gradients | B3, T2, P5, P7, P9 | Real `JtV` in the harness; matches P4 ground truth; Revolve implemented |
| M6 — Solver-in-the-loop | F4, F5 (F6 contingent) | Multi-step BPTT beats the M4 baseline |

M3 runs parallel to M2. P1–P4 need only `forward()` and the state getters, both already bound, so the harness can be built against `jw_baroclinic` ICs while E3 is in progress.

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 6h BPTT window infeasible at ne30 (~100 GB naive) | High | High | P5 Revolve; P8 alternatives |
| ne256 BPTT out of reach entirely | Med-High | High | P7 extrapolates early |
| `Tape` capacity too small for useful per-step depth at ne30 | Med | High | P9 early; P5 layers above it |
| Missing `dp` factor in the θv head — plausible model, wrong by ~10³ | Med | High | D2 contract; unit test vs. analytic tendency |
| Vanishing gradients where the tracer limiter clips | Med-High | Med | D4 penalty; raise with C++ team |
| Model trained on one tracer ordering applied to another | Med | High | B11 registry persisted with checkpoint |
| Per-step state copy dominates (~1.4 GB/call at ne30) | Med | High | B8 measures early |
| PyTorch-native optimizers (Adam/LBFGS) insufficient for M6 — convergence robustness or ROL's distributed vector support turns out necessary at ne120+/ne256 (§5.3.4) | Med | Med-High | F5 as interim step up from Adam; F6 revisits PyROL, A9 tracks the toolchain prerequisite |
| ERA5→theta-l NH initialization shock | Med | Med | E5 measures; consider nudged spin-up |
| Building against unmerged, history-rewritten branches | Med | Med | Consume merged `master` only |
| ne2 measurements mislead the ne30 design | High if unaddressed | High | P3 sweeps at ne30; T5 documents the limit |

---

## 9. Open questions

1. **Mass conservation, in which sense?** Dry-air mass is safe. NN tracer forcing moves `ps` and `dp3d` via `FQps`. Is the requirement "no spurious dry-mass source" — satisfied, nothing to do — or "`ps` must not drift", needing a global constraint on the NN's `qv` output or restriction to non-water tracers?
2. **`Tape` capacity** — what per-step differentiation depth is achievable at ne30, and how does it surface to Python?
3. **`m_ft`** — live or vestigial in the states path?
4. **Is the 6-hour interval the training window**, or a supervision interval with shorter BPTT inside it? The central research question; P6 answers it.
5. **Does M6 need ROL's parallel vector support?** BPTT stops being single-node at ne120 (§5.3.4). F4's distributed parameter-handling note should answer this independently of whether LBFGS (F5) is numerically sufficient — the two are separate reasons F6 might trigger.

---

## 10. Immediate next steps

1. B1 — forcing binding; gates M1.
2. Send §4 to the C++ team: `ForcingFunctor` genericity, `JtV` signature, `Tape` capacity, limiter kink. Ask the expected merge window for `rk-adjoint-stepping`.
3. B10 + B11 — clarify `ft`, design the tracer registry.
4. P1 + P2 — harness with stub backward, parallel to B1, on `jw_baroclinic` ICs.
5. A8 — ne30 configuration; needed before any P3 measurement means anything.
6. F3 — Adam-based training loop against `forward()` (PyTorch-native optimizers first, §2), early and parallel.