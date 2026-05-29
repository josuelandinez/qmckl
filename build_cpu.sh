#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "=========================================================="
echo " Phase 1: Generating QMCkl Source Files (Emacs/Autotools) "
echo "=========================================================="
./autogen.sh
# We disable the legacy Fortran/Python wrappers to speed up configuration
./configure --disable-fortran --disable-python
# Tangle the .org files into the src/ directory
make -j 4 

echo "=========================================================="
echo " Phase 2: Compiling Core HPC Architecture (CMake)         "
echo "=========================================================="
mkdir -p build_cmake
cd build_cmake
rm -rf * # Ensure a clean CMake cache

# Use the TREXIO_PREFIX environment variable if set, otherwise default to the local install
TREXIO_PREFIX=${TREXIO_PREFIX:-$HOME/Codes/QMCkl_Kokkos/install}

cmake -DCMAKE_PREFIX_PATH=$TREXIO_PREFIX ..
make -j 4

echo "=========================================================="
echo " Phase 3: Executing Baseline Validation Test              "
echo "=========================================================="
export LD_LIBRARY_PATH=$TREXIO_PREFIX/lib:$PWD:$LD_LIBRARY_PATH
./run_test

echo "=========================================================="
echo " SUCCESS: Baseline Compilation and Test Complete!         "
echo "=========================================================="
