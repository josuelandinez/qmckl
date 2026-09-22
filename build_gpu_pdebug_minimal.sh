#!/bin/bash
set -e 

module --force purge
module load Stages/2026 GCC/14.3.0 OpenMPI/5.0.8 CUDA/13 CMake/3.31.8 Doxygen/1.14.0 OpenBLAS/0.3.30 Autotools/20250527 HDF5/1.14.6-serial

# Fresh, separate build directory.
rm -rf build_gpu_pdebug_minimal && mkdir build_gpu_pdebug_minimal && cd build_gpu_pdebug_minimal

TREXIO_PREFIX=$PWD/../../install

# Same two changes as the CPU minimal build, applied to the host-side flags:
# -UNDEBUG (countering the same Release+manual-CFLAGS NDEBUG-stacking issue -
# see build_cpu_pdebug_minimal.sh for the full explanation) and -march=znver2
# removed. --use_fast_math is already absent from CMAKE_CUDA_FLAGS in the
# current build script, so no change needed there - if the bug persists here,
# that hypothesis is independently ruled out too, not just untested.
export CFLAGS="-O3 -fPIC -g -UNDEBUG"
export CXXFLAGS="-O3 -fPIC -g -UNDEBUG"

cmake .. \
      -DENABLE_KOKKOS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkosKernels_ENABLE_TPL_CUBLAS=ON \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      -DKokkos_ENABLE_PROFILING=ON \
      -DKokkos_ENABLE_PROFILING_LOAD_PRINT=ON \
      -DKokkos_ENABLE_CUDA_UVM=OFF \
      -DENABLE_KTUNE=ON \
      -DCMAKE_CUDA_FLAGS="-O3 -lineinfo -UNDEBUG" \
      -DTREXIO_INCLUDE_DIR=$TREXIO_PREFIX/include \
      -DTREXIO_LIBRARY=$TREXIO_PREFIX/lib64/libtrexio.so

make verify_orbitals kp_nvtx_connector -j 8

export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib64:$LD_LIBRARY_PATH
export KOKKOS_PROFILE_LIBRARY=$PWD/_deps/kokkos_tools-build/profiling/nvtx-connector/libkp_nvtx_connector.so

echo ""
echo "=========================================================="
echo " Minimal-flags GPU library build complete: $PWD            "
echo " Next: rebuild the benchmark harness against this library  "
echo " (see build_qmckl_bench_minimal.sh)                        "
echo "=========================================================="
