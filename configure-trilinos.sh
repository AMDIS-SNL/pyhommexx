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