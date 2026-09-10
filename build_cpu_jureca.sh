#!/bin/bash

set -e 

module --force purge

module load Stages/2026 \
       GCC/14.3.0 \
       OpenMPI/5.0.8 \
       CUDA/13 \
       CMake/3.31.8 \
       Doxygen/1.14.0 \
       OpenBLAS/0.3.30 \
       Autotools/20250527 \
       HDF5/1.14.6-serial


echo "=========================================================="
echo " Compiling Kokkos CPU (OpenMP) Architecture               "
echo "=========================================================="
rm -rf build_cpu
mkdir -p build_cpu && cd build_cpu

TREXIO_PREFIX=$PWD/../../install

echo "TREXIO: $TREXIO_PREFIX"

# Check that TREXIO exists
test -f "$TREXIO_PREFIX/include/trexio.h"
test -f "$TREXIO_PREFIX/lib64/libtrexio.so"



cmake .. -DENABLE_KOKKOS=ON \
         -DKokkos_ARCH_ZEN2=ON \
         -DKokkos_ENABLE_OPENMP=ON \
         -DKokkos_ENABLE_CUDA=OFF \
         -DBUILD_TESTING=ON \
	 -DCMAKE_BUILD_TYPE=Release \
         -DKokkosKernels_ENABLE_SPARSE=OFF \
         -DKokkosKernels_ENABLE_GRAPH=OFF \
         -DKokkosKernels_ENABLE_BATCHED=OFF \
         -DKokkosKernels_ENABLE_ODE=OFF \
         -DKokkosKernels_ENABLE_LAPACK=OFF \
	 -DTREXIO_INCLUDE_DIR="$TREXIO_PREFIX/include" \
         -DTREXIO_LIBRARY=$TREXIO_PREFIX/lib64/libtrexio.so

make -j 4

echo "=========================================================="
echo " Executing Kokkos CPU Validation                          "
echo "=========================================================="

export LD_LIBRARY_PATH=$TREXIO_PREFIX/lib64:$LD_LIBRARY_PATH    

# We link dynamically against the built library using the correct relative paths
gcc -I../include -I../src -I$TREXIO_PREFIX/include ../tests/verify_orbitals.c \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -o verify_orbitals_cpu

#source ../tests/testvenv/bin/activate


export LD_LIBRARY_PATH=$PWD:build_cpu:$LD_LIBRARY_PATH

./verify_orbitals_cpu ../files_qmckl_kokkos/water.hdf5


echo "CPU Build complete. Executable is in build_cpu/"
