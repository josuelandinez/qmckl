#!/bin/bash
set -e 

echo "=========================================================="
echo " Compiling Kokkos GPU (CUDA) Architecture                 "
echo "=========================================================="
mkdir -p build_gpu && cd build_gpu
rm -rf * 

TREXIO_PREFIX=$HOME/Codes/QMCkl_Kokkos/install

cmake .. -DENABLE_KOKKOS=ON \
         -DKokkos_ENABLE_OPENMP=OFF \
         -DKokkos_ENABLE_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=86 \
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
echo " Executing Kokkos GPU Validation                          "
echo "=========================================================="
export LD_LIBRARY_PATH=$PWD:build_gpu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH    

gcc -Iinclude -Isrc -I$HOME/Codes/QMCkl_Kokkos/install/include tests/verify_orbitals.c \
    -Lbuild_gpu -L$HOME/Codes/QMCkl_Kokkos/install/lib \
    -lqmckl -ltrexio -lm -lstdc++ -o verify_orbitals_gpu

source ${PWD}/testvenv/bin/active


./verify_orbitals_gpu ../water.hdf5
