# AMDIS Environment Setup 

## At every login:
- Load the modules:  `. /projects/amdis/env/amdis-modules.sh`.
- Define a scratch directory: `export SCRATCH=/gpfs/pabosle`


## Once per platform
- Create a virtual environment in your scratch space:
  ```
  python -m venv $SCRATCH/amdis-venv
  . $SCRATCH/amdis-env/bin/activate
  python -m pip install --upgrade pip
  python -m install -r /projects/amdis/env/requirements.txt
  ```
  
## Build HommeXX

Activate the virtual environment: `. $SCRATCH/amdis-env/bin/activate`.

Run `/projects/amdis/env/configure-pyhommexx.sh`.

The script mostly works to set up a CPU build on a cee-compute server.
13-AUG-2026: Some CMake warnings are triggered, but they can be ignored.

Build it: `cmake --build $SCRATCH/build-e3sm-amdis -j 48`.

For the DCMIP-2016 test 1 dataset workflow (E9/E10/D8) there are two additional
builds, both separate from the primary `nlev=128` pyhommexx build:

- `configure-homme-dcmip-nlev30.sh` — Fortran HommeXX at nlev=30. Builds the
  `theta-l-nlev30-native` test executable that generates the training data (E9).
  Build dir: `$SCRATCH/build-homme-dcmip-nlev30/`. After
  `cmake --build ... && cmake --install ...`, `cd` into the mirrored
  `dcmip_tests/dcmip2016_test1_baroclinic_wave/theta-l/` and submit the jobscript.
- `configure-pyhommexx-nlev30.sh` — pyhommexx shared library at nlev=30 for the
  D8 student. Build dir: `$SCRATCH/build-e3sm-amdis-nlev30/`. Select at runtime
  via `export PYHOMMEXX_LIB_PATH=$SCRATCH/build-e3sm-amdis-nlev30/src/theta-l_kokkos/pyhommexx/`.

The primary `nlev=128` pyhommexx build stays untouched for the ERA5 workflow.

### Check the build

At this point, HommeXX (the C++ side) should be working.  Let's make sure:

```
cd $SCRATCH/build-e3sm-amdis
ctest -R sacado -V
```

Success:

```
The following tests passed:
	caar_sacado_ut
	eos_sacado_ut
	dirk_sacado_ut
	tridiag_sacado_ut
	sacado_ut_test

100% tests passed, 0 tests failed out of 5

Label Time Summary:
AMDIS    = 132.36 sec*proc (4 tests)
unit     =   0.76 sec*proc (1 test)
```


# Prerequisites

Items in this section are required to run PyHommeXX, but need to be done only once per platform.

**Note** that changing the Kokkos device type might count as a different platform.

## Trilinos/Sacado

Currently installed at `/projects/amdis/tpl`.

Assuming the modules are loaded with `. /projects/amdis/env/amdis-modules.sh`, Sacado can be built as:
```
cd /projects/amdis/tpl
tar -xf Trilinos-trilinos-release-16-2-0.tar
cd /projects/amdis/env
./configure-trilinos.sh
cmake --build $SCRATCH/build-sacado -j
cmake --install $SCRATCH/build-sacado --prefix=/projects/amdis/tpl
```

Where `configure-trilinos.sh` is this script:
```
TRILINOS_SRC=/projects/amdis/tpl/Trilinos-trilinos-release-16-2-0

cmake -B $SCRATCH/build-sacado \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DTPL_ENABLE_MPI=ON \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpicxx \
-DTrilinos_ENABLE_Fortran=OFF \
-DTrilinos_ENABLE_OpenMP=ON \
-DTrilinos_ENABLE_ALL_PACKAGES=OFF \
-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
-DTrilinos_ENABLE_Sacado=ON \
-DTrilinos_ENABLE_TESTS=OFF \
${TRILINOS_SRC}
```


## NetCDF Fortran

Currently installed at `/projects/amdis/tpl`.

The CEE modules only provide NetCDF C libraries.   We have to build our own NetCDF Fortran, which Homme requires.  Assuming the modules are loaded with `. /projects/amdis/env/amdis-modules.sh`, NetCDF Fortran version 4.6.2 can be built with:

```
cd /projects/amdis/tpl
tar -xf netcdf-fortran-4.6.2.tar
cd netcdf-fortran-4.6.2
./configure --prefix=/projects/amdis/tpl CPPFLAGS="-I$HDF5_ROOT/include -I$NETCDF_ROOT/include" LDFLAGS="-L$HDF5_ROOT/lib -L$NETCDF_ROOT/lib" CC=mpicc FC=mpifort
make -j && make install
```

