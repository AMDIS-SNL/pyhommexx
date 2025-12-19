# PyHomme: a mini app to run hommexx from python

This mini app shows how to use the newly developed pyhommexx library
to call hommexx from python. It is a very small example, which does
very little, but it serves as a starting point.

In order to use the theta-nlev128.py executable, you must

- have numpy, xarray, mpi4py installed (via pip is enough)
- have an MPI library available in your PATH/LD_LIBRARY_PATH
- edit PYHOMMEXX_LIB_PATH to point to the fodler where pyhommexx was built
