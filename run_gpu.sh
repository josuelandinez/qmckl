#!/bin/bash
set -e

echo "=========================================================="
echo " Running GPU (CUDA) Verification Benchmark"
echo "=========================================================="

# Define library paths
TREXIO_PREFIX=${TREXIO_PREFIX:-$HOME/Codes/QMCkl_Kokkos/install}
export LD_LIBRARY_PATH=$TREXIO_PREFIX/lib:$PWD/build_kokkos_gpu:$LD_LIBRARY_PATH

# Isolate the primary GPU
export CUDA_VISIBLE_DEVICES=0

TARGET_FILE="water.hdf5"

if [ ! -f "$TARGET_FILE" ]; then
    echo "CRITICAL: $TARGET_FILE not found in the current directory."
    exit 1
fi

# Execute the GPU build
./build_kokkos_gpu/verify_orbitals $TARGET_FILE
