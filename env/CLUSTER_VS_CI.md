# CI vs. cluster environment

Task A6. Scope: document where the GitHub Actions CI environment
(`.github/workflows/pyhommexx-ci.yml`) diverges from the SNL cee-compute
cluster environment (`env/README.md`, `env/amdis-modules.sh`,
`env/configure-pyhommexx.sh`), and what that divergence means for what CI
actually validates. No build-recipe changes here — that's separate work if
the gaps below turn out to need closing.

## Comparison

| Aspect | CI (`ubuntu-24.04` GitHub runner) | Cluster (cee-compute, via `amdis-modules.sh`) |
| --- | --- | --- |
| Toolchain source | apt packages | environment modules (`aue/*`) |
| gcc/g++/gfortran | apt `gcc-12` → **12.4.0** (apt candidate, checked 2026‑08‑17 on Ubuntu 24.04.4 LTS) | module `aue/gcc/12.3.0` |
| OpenMPI | apt `libopenmpi-dev` → **4.1.6-7ubuntu2** | module `aue/openmpi/4.1.6-...` — matches |
| HDF5 | apt `libhdf5-openmpi-dev` → **1.10.10** | module `aue/hdf5/1.14.3-...` — **major-version gap (1.10 vs 1.14)** |
| NetCDF-C | apt `libnetcdf-dev` → **4.9.2-5ubuntu4** | module `aue/netcdf-c/4.9.2-...` — matches |
| NetCDF-Fortran | apt `libnetcdff-dev` → packaged as NetCDF-Fortran **4.5.4** (Ubuntu labels it `4.6.0+really4.5.4`) | built from source, **4.6.2**, installed to `/projects/amdis/tpl` (the cluster's modules don't ship a Fortran build at all, per `env/README.md`) |
| CMake | pip `cmake==3.31.6` | module `aue/cmake/3.31.6` — matches |
| Trilinos/Sacado | built from source on cache miss, tag `trilinos-release-16-2-0`, installed under the ephemeral per-run `$GITHUB_WORKSPACE/_deps` | built once by hand, same tag, installed to the shared, persistent `/projects/amdis/tpl` |
| Kokkos backend | Serial + OpenMP; `Kokkos_ENABLE_CUDA=OFF` (no GPU on hosted runners) | Serial + OpenMP; `Kokkos_ENABLE_CUDA=OFF` **in the current `configure-pyhommexx.sh`**, even though the cluster loads a CUDA module (`aue/cuda/12.4.0`) and presumably has GPU-capable nodes |
| `Kokkos_ARCH_NATIVE` | ON, tuned to whatever CPU the GitHub runner assigns that run | ON, tuned to the cee-compute node's CPU — a "native" build from one side isn't portable to the other |
| `CMAKE_BUILD_TYPE` | `RelWithDebInfo` by default (`DEBUG`/`Release` selectable via `workflow_dispatch`) | hardcoded `DEBUG` in `configure-pyhommexx.sh` |
| `HOMME_BUILD_EXECS` | explicitly `OFF` — build is bounded to the sacado tests + the pyhommexx module | not set — uses whatever HommeXX's own CMake default is (unverified here; if it defaults ON, the cluster build has a larger surface than CI's) |
| E3SM source | fetched fresh each run from `AMDIS-SNL/E3SM@master` (or a `workflow_dispatch` ref) into `./e3sm` | staged once at `/projects/amdis/e3sm-amdis/components/homme`; how/when it's refreshed isn't documented |
| Python | `actions/setup-python` 3.13, packages installed straight into the runner, no venv | module `aue/python/3.13.2` + a venv the user creates by hand under `$SCRATCH` |
| Python deps installed | only `cmake`, `nanobind==2.11.0`, `numpy`, `mpi4py` — just enough for the smoke import | full `env/requirements.txt`: adds Cartopy, matplotlib, pandas, scipy, xarray, netcdf4, and CUDA wheels of torch/torchvision/torchaudio |
| MPI rank count | oversubscribe forced on (`OMPI_MCA_rmaps_base_oversubscribe=1`) for a 2–4 core runner | not set; runs on real allocated cores |
| What's actually tested | `ctest -R sacado` (gating) + a `continue-on-error` Python smoke import | `ctest -R sacado -V` per `env/README.md`; no documented Python-level test |
| `PYHOMMEXX_NLEV` / `PYHOMMEXX_QSIZE` | 128 / 10 | 128 / 10 — matches |

Package-version cells above are drawn from `apt-cache policy` output on a
Ubuntu 24.04.4 LTS host, i.e. the same distro/release as the GitHub-hosted
`ubuntu-24.04` runner — this is a primary-source check I ran directly, not a
citation to an external document, and it's a point-in-time read: Ubuntu's
archive updates, so exact patch versions will drift. Cluster-side module
versions are read directly from `env/amdis-modules.sh` and `env/README.md`
in this repo. I'm ~90% confident in the apt version numbers as of today;
I have not verified the cee-compute host OS or HommeXX's CMake default for
`HOMME_BUILD_EXECS`, and flag both as open items below rather than guessing.

## Notes

- NN/data-pipeline Python (`torch`, `xarray`, `Cartopy`, ...) isn't
  installed in CI — only `numpy`/`mpi4py` are. Expected, not a gap: that
  code doesn't exist yet (D., E. in §6). Worth adding to CI once it lands;
  no reason to pay for it now while it'd only slow the build down.
- HDF5 (1.10 vs 1.14) and NetCDF-Fortran (4.5.4 vs 4.6.2) are on different
  versions between CI and cluster.
- CUDA is off on both sides today. CI structurally can't turn it on — no
  GPU on hosted runners — so this only becomes relevant once
  `Kokkos_ENABLE_CUDA=ON` work actually starts.
- Build type differs: `DEBUG` on the cluster vs `RelWithDebInfo` (CI
  default).
- Trilinos install lifecycle differs: CI rebuilds on cache miss inside an
  ephemeral per-run workspace; the cluster's install is a long-lived,
  shared install under `/projects/amdis/tpl`.
- `HOMME_BUILD_EXECS` may differ in scope (CI sets it explicitly `OFF`;
  cluster leaves it unset) — unverified what HommeXX's CMake default is.

## Unverified / follow-up

- Cee-compute host OS and its exact package/module versions beyond what
  `amdis-modules.sh` names.
- HommeXX's CMake default for `HOMME_BUILD_EXECS` when unset (affects point
  6 above).
- Update cadence for the staged E3SM source at
  `/projects/amdis/e3sm-amdis`.
- Whether the cluster's CUDA module is actually exercised by anyone today,
  or purely available-but-unused, as it currently is in `configure-pyhommexx.sh`.