TRILINOS_SRC=/projects/amdis/tpl/Trilinos

cmake -B $SCRATCH/build-trilinos \
-S $TRILINOS_SRC \
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
