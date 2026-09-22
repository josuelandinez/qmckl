#!/bin/bash
set -e 

module --force purge
module load Stages/2026 GCC/14.3.0 OpenMPI/5.0.8 CUDA/13 CMake/3.31.8 Doxygen/1.14.0 OpenBLAS/0.3.30 Autotools/20250527 HDF5/1.14.6-serial likwid/5.4.1

# Fresh, separate build directory - does not touch build_cpu_pdebug or any
# other build you have.
rm -rf build_cpu_pdebug_minimal && mkdir build_cpu_pdebug_minimal && cd build_cpu_pdebug_minimal

TREXIO_PREFIX=$PWD/../../install

export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

# Three changes from the current build script, all suspects tested together
# in one build rather than one rebuild cycle per flag, given how much time
# each cycle has cost so far. If this resolves the segfault, re-add flags one
# at a time afterward to identify which one specifically mattered.
#
# 1. -UNDEBUG added: CMAKE_BUILD_TYPE=Release appends its own default flags
#    (-O3 -DNDEBUG for GCC) AFTER whatever CFLAGS/CXXFLAGS are already set
#    from the environment - a documented CMake behavior, not a guess. That
#    means -DNDEBUG has very likely been silently active in every build so
#    far, disabling every assert() in the codebase (qmckl_context.c alone has
#    several, right after each init step). If an init step has been quietly
#    failing for large problem sizes, NDEBUG would hide it completely rather
#    than stopping execution with a clear error - -UNDEBUG re-enables asserts
#    regardless of what Release's own defaults add.
# 2. -march=znver2 removed: aggressive, architecture-specific vectorization
#    (AVX2/FMA) - untested in isolation until now as a cause of the
#    non-deterministic, size-correlated crash.
# 3. -fno-trapping-math removed: relaxes IEEE floating-point exception
#    semantics - kept in the original build, removing here for the cleanest
#    possible comparison against native, which does not use this flag.
export CFLAGS="-O3 -fopenmp -fPIC -g -UNDEBUG"
export CXXFLAGS="-O3 -fopenmp -fPIC -g -UNDEBUG"
export LDFLAGS="-fopenmp -g"

cmake .. \
      -DENABLE_KOKKOS=ON \
      -DKokkos_ARCH_ZEN2=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkosKernels_ENABLE_TPL_BLAS=ON \
      -DBLA_VENDOR=OpenBLAS \
      -DBLAS_LIBRARIES=openblas \
      -DBLAS_LIBRARY_DIRS="$EBROOTOPENBLAS/lib64" \
      -DKokkos_ENABLE_PROFILING=ON \
      -DKokkos_ENABLE_PROFILING_LOAD_PRINT=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_KTUNE=ON \
      -DTREXIO_INCLUDE_DIR="$TREXIO_PREFIX/include" \
      -DTREXIO_LIBRARY="$TREXIO_PREFIX/lib64/libtrexio.so"
      # ENABLE_KTUNE left ON, unchanged - this builds whatever
      # qmckl_orbitals_kokkos.cpp is currently in the source tree (the
      # no-KTune diagnostic version already swapped in, per earlier testing,
      # which was already shown NOT to be the cause on its own).

make verify_orbitals kp_kernel_timer -j 8

export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib64:$LD_LIBRARY_PATH
export KOKKOS_PROFILE_LIBRARY=$PWD/_deps/kokkos_tools-build/profiling/kernel-timer/libkp_kernel_timer.so

echo ""
echo "=========================================================="
echo " Minimal-flags CPU library build complete: $PWD            "
echo " Next: rebuild the benchmark harness against this library  "
echo " (see build_qmckl_bench_minimal.sh)                        "
echo "=========================================================="
