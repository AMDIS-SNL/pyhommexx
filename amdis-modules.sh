module purge
module load aue/gcc/12.3.0
module load aue/cuda/12.4.0-gcc-12.3.0
module load aue/openmpi/4.1.6-gcc-12.3.0-cuda-12.4.0
module load aue/hdf5/1.14.3-gcc-12.3.0-openmpi-4.1.6
module load aue/netcdf-c/4.9.2-gcc-12.3.0-openmpi-4.1.6
module load aue/cmake/3.31.6
module load aue/binutils/2.45
module load aue/python/3.13.2
export LD_LIBRARY_PATH="$NETCDF_ROOT/lib:$HDF5_ROOT/lib:$LD_LIBRARY_PATH"

