#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "=========================================================="
echo " Compiling Native C Architecture (CMake Only)             "
echo "=========================================================="
mkdir -p build_native && cd build_native
rm -rf * # Ensure pristine cache

TREXIO_PREFIX=$HOME/Codes/QMCkl_Kokkos/install

cmake .. -DENABLE_KOKKOS=OFF \
         -DBUILD_TESTING=ON \
         -DTREXIO_INCLUDE_DIR=$TREXIO_PREFIX/include \
         -DTREXIO_LIBRARY=$TREXIO_PREFIX/lib/libtrexio.so

make -j 4

echo "=========================================================="
echo " Executing Native Baseline Validation                     "
echo "=========================================================="
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH
./verify_orbitals ../water.hdf5
