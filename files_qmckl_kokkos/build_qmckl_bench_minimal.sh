#!/bin/bash
set -e

module --force purge
module load Stages/2026 GCC/14.3.0 OpenMPI/5.0.8 CUDA/13 CMake/3.31.8 Doxygen/1.14.0 OpenBLAS/0.3.30 Autotools/20250527 HDF5/1.14.6-serial likwid/5.4.1

BASE_DIR="/p/project1/numeriqs/landinezborda1/B2/QMCkl_kokkos"
TREXIO_PREFIX="${BASE_DIR}/install"
export OMP_STACKSIZE=1G

# Only CPU Kokkos and GPU Kokkos - native is untouched (it's never shown the
# segfault/NaN and isn't part of this comparison), and this focuses rebuild
# time on the two configurations actually under test.
SRC_DIRS=(
    "${BASE_DIR}/qmckl_test"
    "${BASE_DIR}/qmckl_test"
)
LIB_DIRS=(
    "${BASE_DIR}/qmckl_test/build_cpu_pdebug_minimal"
    "${BASE_DIR}/qmckl_test/build_gpu_pdebug_minimal"
)
BUILD_DIRS=(
    "build_cpu_kokkos_minimal"
    "build_gpu_kokkos_minimal"
)
# -UNDEBUG added for the same reason as the library builds; -march=znver2
# removed for the CPU harness to match the library's own flags.
C_FLAGS=(
    "-O3 -fopenmp -UNDEBUG"
    "-O3 -UNDEBUG"
)

for i in "${!BUILD_DIRS[@]}"; do
    SRC_DIR="${SRC_DIRS[$i]}"
    LIB_DIR="${LIB_DIRS[$i]}"
    BUILD_DIR="${BUILD_DIRS[$i]}"
    CURRENT_CFLAGS="${C_FLAGS[$i]}"

    echo "=========================================================="
    echo " Building Benchmarks: ${BUILD_DIR}"
    echo " QMCKL_LIBRARY      : ${LIB_DIR}/libqmckl.so"
    echo " CMAKE_C_FLAGS      : ${CURRENT_CFLAGS}"
    echo "=========================================================="

    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    cmake .. \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_C_FLAGS="${CURRENT_CFLAGS}" \
	  -DTREXIO_INCLUDE_DIR="${TREXIO_PREFIX}/include" \
	  -DTREXIO_LIBRARY="${TREXIO_PREFIX}/lib64/libtrexio.so" \
	  -DQMCKL_INCLUDE_DIR="${SRC_DIR}/include" \
	  -DQMCKL_LIBRARY="${LIB_DIR}/libqmckl.so"

    make -j 8

    cd ..
    echo "Successfully built ${BUILD_DIR}."
    echo ""
done
