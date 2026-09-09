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
         -DBUILD_TESTING=OFF \
         -DKokkos_ENABLE_TESTS=OFF \
         -DKokkosKernels_ENABLE_TESTS=OFF \
         -DKokkos_ENABLE_EXAMPLES=OFF \
         -DKokkosKernels_ENABLE_EXAMPLES=OFF \
         -DKokkosKernels_ENABLE_TPL_CUBLAS=ON \
         -DKokkosKernels_ENABLE_SPARSE=OFF \
         -DKokkosKernels_ENABLE_GRAPH=OFF \
         -DKokkosKernels_ENABLE_BATCHED=OFF \
         -DKokkosKernels_ENABLE_ODE=OFF \
         -DKokkosKernels_ENABLE_LAPACK=OFF \
         -DTREXIO_INCLUDE_DIR=$TREXIO_PREFIX/include \
         -DTREXIO_LIBRARY=$TREXIO_PREFIX/lib/libtrexio.so

# Using more threads for compilation significantly speeds up template instantiation
make -j 4

echo "=========================================================="
echo " Executing Kokkos GPU Validation                          "
echo "=========================================================="
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH    

# We link dynamically against the built library using the correct relative paths

# We link dynamically against the built library using the correct relative paths
gcc -I../include -I../src -I$TREXIO_PREFIX/include ../tests/verify_orbitals.c \
    -L. -L$TREXIO_PREFIX/lib \
    -lqmckl -ltrexio -lm -lstdc++ -o verify_orbitals_gpu

###source ../tests/testvenv/bin/activate

export LD_LIBRARY_PATH=$PWD:build_gpu:$LD_LIBRARY_PATH


./verify_orbitals_gpu ../files_qmckl_kokkos/water.hdf5


