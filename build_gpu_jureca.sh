#!/bin/bash
set -e 

module --force purge

module load Stages/2026 \
       GCC/14.3.0 \
       OpenMPI/5.0.8 \
       CUDA/13 \
       CMake/3.31.8 \
       Doxygen/1.14.0 \
       OpenBLAS/0.3.30 \
       Autotools/20250527 \
       HDF5/1.14.6-serial


echo "=========================================================="
echo " Compiling Kokkos GPU (CUDA) Architecture                 "
echo "=========================================================="
rm -rf build_gpu
mkdir build_gpu
cd build_gpu


TREXIO_PREFIX=$PWD/../../install


cmake .. \
      -DENABLE_KOKKOS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_OPENMP=OFF \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
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
      -DKokkos_ENABLE_CUDA_UVM=OFF \
      -DTREXIO_INCLUDE_DIR=$TREXIO_PREFIX/include \
      -DTREXIO_LIBRARY=$TREXIO_PREFIX/lib64/libtrexio.so

# Using more threads for compilation significantly speeds up template instantiation
make -j 4

echo "=========================================================="
echo " Executing Kokkos GPU Validation                          "
echo "=========================================================="
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib64:$LD_LIBRARY_PATH    

# We link dynamically against the built library using the correct relative paths

# We link dynamically against the built library using the correct relative paths
gcc -I../include -I../src -I$TREXIO_PREFIX/include ../tests/verify_orbitals.c \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -o verify_orbitals_gpu

###source ../tests/testvenv/bin/activate

export LD_LIBRARY_PATH=$PWD:build_gpu:$LD_LIBRARY_PATH


./verify_orbitals_gpu ../files_qmckl_kokkos/water.hdf5


