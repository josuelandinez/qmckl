#!/bin/bash
set -e 

echo "=========================================================="
echo " Compiling Kokkos CPU (OpenMP) Architecture               "
echo "=========================================================="
mkdir -p build_cpu && cd build_cpu
rm -rf * 

TREXIO_PREFIX=$HOME/Codes/QMCkl_Kokkos/install

cmake .. -DENABLE_KOKKOS=ON \
         -DKokkos_ENABLE_OPENMP=ON \
         -DKokkos_ENABLE_CUDA=OFF \
         -DBUILD_TESTING=ON \
         -DKokkosKernels_ENABLE_SPARSE=OFF \
         -DKokkosKernels_ENABLE_GRAPH=OFF \
         -DKokkosKernels_ENABLE_BATCHED=OFF \
         -DKokkosKernels_ENABLE_ODE=OFF \
         -DKokkosKernels_ENABLE_LAPACK=OFF \
         -DTREXIO_INCLUDE_DIR=$TREXIO_PREFIX/include \
         -DTREXIO_LIBRARY=$TREXIO_PREFIX/lib/libtrexio.so

make -j 4

echo "=========================================================="
echo " Executing Kokkos CPU Validation                          "
echo "=========================================================="


gcc -Iinclude -Isrc -I$HOME/Codes/QMCkl_Kokkos/install/include tests/verify_orbitals.c \
    -Lbuild_cpu -L$HOME/Codes/QMCkl_Kokkos/install/lib \
    -lqmckl -ltrexio -lm -lstdc++ -o verify_orbitals

source ${PWD}/testvenv/bin/active

export LD_LIBRARY_PATH=$PWD:build_cpu:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH    

./verify_orbitals ../water.hdf5
