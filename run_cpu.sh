#!/bin/bash
set -e

echo "=========================================================="
echo " Running CPU (OpenMP) Verification Benchmark"
echo "=========================================================="

# Define library paths
TREXIO_PREFIX=${TREXIO_PREFIX:-$HOME/Codes/QMCkl_Kokkos/install}
export LD_LIBRARY_PATH=$TREXIO_PREFIX/lib:$PWD/build_cmake:$LD_LIBRARY_PATH

# OpenMP Thread Binding Optimization for Kokkos
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}

TARGET_FILE="/home/siegfried/Codes/QMCkl_Kokkos/trexio_tools/data/methane_sphe.hdf5"

if [ ! -f "$TARGET_FILE" ]; then
    echo "CRITICAL: $TARGET_FILE not found in the current directory."
    exit 1
fi

# Execute the CPU build
./build_cmake/verify_orbitals $TARGET_FILE
