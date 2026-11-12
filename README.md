# PyHomme: a mini app to run hommexx from python

[![CI](https://github.com/AMDIS-SNL/pyhommexx/actions/workflows/pyhommexx-ci.yml/badge.svg?branch=main)](https://github.com/AMDIS-SNL/pyhommexx/actions/workflows/pyhommexx-ci.yml)

This mini app shows how to use the newly developed pyhommexx library
to call hommexx from python. It is a very small example, which does
very little, but it serves as a starting point.

This repository contains the Python side of our work.  The C++ side is contained within the [AMDIS-SNL E3SM fork](git@github.com:AMDIS-SNL/E3SM.git).

In order to use the scripts in this folder, you must

- have a python environment set up (nanobind, numpy, xarray, mpi4py, etc.) installed via pip is ok.
- have an MPI library available in your PATH/LD_LIBRARY_PATH
- edit PYHOMMEXX_LIB_PATH to point to the fodler where pyhommexx was built

In order for homme to build pyhommexx, you need some extra cmake settings, such as:

  -D HOMME_BUILD_PYHOMMEXX:BOOL=ON                          \
  -D PYHOMMEXX_NLEV:STRING=128                              \
  -D PYHOMMEXX_QSIZE:STRING=10                              \

The second and third can be omitted if you are ok with 128 lev and qsize_d=10,
as those are already the default ones. If you ONLY care about using pyhommexx,
you could also turn off other things so they don't get built:

  -D HOMME_AMDIS_PROJECT:BOOL=ON                            \
  -D HOMMEXX_ENABLE_FAD_TYPES:BOOL=ON                       \
  -D HOMMEXX_BFB_TESTING:BOOL=OFF                           \
  -D HOMME_BUILD_EXECS:BOOL=OFF                             \
  
## Tests

The scripts in `pyhomme/` that assert something are run by the CI (see `.github/workflows/pyhommexx-ci.yml`).
They need `PYHOMMEXX_LIB_PATH` to point to the folder where pyhommexx was built, and must be run from `pyhomme/`
(the namelists use relative paths for the vertical coordinate files):

- `demo_forcing.py -nml namelist.nl`: applies a dynamics forcing, and checks the effect on the state.
- `test_prim_step_adjoint.py -nml namelist_adjoint.nl`: checks that the forward sensitivities of `prim_step`
  (run with the `dpfad` scalar type) match its adjoint (`prim_step_adj`), field by field, with the dot-product test
  `<dF*du0, lamN> == <du0, dF^T*lamN>`. It needs homme built with `-D HOMMEXX_ENABLE_FAD_TYPES:BOOL=ON`
  (and without `HOMMEXX_ENABLE_FWD_SENS`, which pyhommexx does not support), and the settings of
  `namelist_adjoint.nl` (`nu_p=0`, `nu_top=0`, `tstep_type=10`, `limiter_option=9`, ...), which are the ones the adjoint supports.

## Timesaver: CEE project space

The prerequisites have already been built and installed to `/projects/amdis/tpl`, using the environment setup files in this [repo's `env/` folder](https://github.com/AMDIS-SNL/pyhommexx/tree/main/env).

To set up PyHomme on a new machine, use the instructions in the [`env/README.md`](https://github.com/AMDIS-SNL/pyhommexx/blob/main/env/README.md) file as a starting point.
