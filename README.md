# PyHomme: a mini app to run hommexx from python

This mini app shows how to use the newly developed pyhommexx library
to call hommexx from python. It is a very small example, which does
very little, but it serves as a starting point.

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
  
## Timesaver: CEE project space

The prerequisites have already been built and installed to `/projects/amdis/tpl`, using the environment setup files in this [repo's `env/` folder](https://github.com/AMDIS-SNL/pyhommexx/tree/main/env).

To set up PyHomme on a new machine, use the instructions in the [`env/README.md`](https://github.com/AMDIS-SNL/pyhommexx/blob/main/env/README.md) file as a starting point.
