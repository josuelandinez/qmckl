#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "=========================================================="
echo " Phase 1: Generating QMCkl Source Files (Emacs/Autotools) "
echo "=========================================================="
./autogen.sh
./configure --disable-fortran --disable-python
make -j 4 

echo "=========================================================="
echo " Phase 2: Compiling Core HPC Architecture (CMake)         "
echo "=========================================================="
mkdir -p build_cmake
cd build_cmake
rm -rf * # Ensure a clean CMake cache

# Use the verified local TREXIO install
TREXIO_PREFIX=${TREXIO_PREFIX:-$HOME/Codes/QMCkl_Kokkos/install}

# Pass TREXIO_PREFIX to CMake and enable the Kokkos CPU OpenMP backend
cmake -DCMAKE_PREFIX_PATH=$TREXIO_PREFIX \
      -DKokkos_ENABLE_OPENMP=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..

make -j 4

echo "=========================================================="
echo " Phase 3: Executing Baseline Validation Test              "
echo "=========================================================="
export LD_LIBRARY_PATH=$TREXIO_PREFIX/lib:$PWD:$LD_LIBRARY_PATH
./run_test

echo "=========================================================="
echo " SUCCESS: Kokkos Integrated and Test Complete!            "
echo "=========================================================="
