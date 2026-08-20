#!/bin/bash
#
# HommeXX Fortran configure for the DCMIP-2016 test 1 training-data run (E9).
# Builds the `theta-l-nlev30` test executable, not the pyhommexx shared lib.
# Sibling: configure-pyhommexx.sh (nlev=128) and configure-pyhommexx-nlev30.sh
#
# After configure:
#   cmake --build $BUILD_DIR -j 48 --target theta-l-nlev30
#   cmake --install $BUILD_DIR       # stages namelists + jobscript into the build tree
#   cd $BUILD_DIR/dcmip_tests/dcmip2016_test1_baroclinic_wave/theta-l
#   sbatch jobscript-dcmip16-hifreq-cee.sh {smoke|prod}
# The exec lands at $BUILD_DIR/test_execs/theta-l-nlev30/theta-l-nlev30 —
# relative path the jobscript expects.
# Perf note: the fork's cmake/SetCompilerFlags.cmake:114 overrides Fortran
# CMAKE_Fortran_FLAGS_RELEASE to "-O0 -g" whenever HOMME_AMDIS_PROJECT is on.
# That's fine for pyhommexx dev iteration but ruinous for a 60-day ne30 run.
# We pass CMAKE_Fortran_FLAGS_RELEASE=-O2 to override that override.

SRC_DIR=/projects/amdis/e3sm-amdis/components/homme
NC_PATH=$NETCDF_ROOT
NF_PATH=/projects/amdis/tpl

BUILD_DIR=$SCRATCH/build-homme-dcmip-nlev30

cmake -B $BUILD_DIR \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_Fortran_COMPILER=mpifort \
-DCMAKE_C_FLAGS="-fPIC" \
-DCMAKE_CXX_FLAGS="-fPIC" \
-DCMAKE_Fortran_FLAGS="-fPIC" \
-DCMAKE_Fortran_FLAGS_RELEASE="-O2" \
-DCMAKE_CXX_STANDARD=17 \
-DHOMME_FIND_BLASLAPACK=ON \
-DHOMMEXX_BFB_TESTING=OFF \
-DHOMME_AMDIS_PROJECT:BOOL=ON \
-DBUILD_HOMME_THETA:BOOL=ON \
-DBUILD_HOMME_THETA_KOKKOS:BOOL=OFF \
-DBUILD_HOMME_PREQX:BOOL=OFF \
-DBUILD_HOMME_PREQX_KOKKOS:BOOL=OFF \
-DBUILD_HOMME_SWEQX:BOOL=OFF \
-DBUILD_HOMME_TOOL:BOOL=OFF \
-DTrilinos_ROOT=${NF_PATH} \
-DNetCDF_C_PATH=$NC_PATH \
-DNetCDF_Fortran_PATH=${NF_PATH} \
-DKokkos_ENABLE_CUDA=OFF \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ARCH_NATIVE=ON \
-S $SRC_DIR
