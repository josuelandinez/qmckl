#!/bin/bash
# Exit immediately if a command exits with a non-zero status
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

TREXIO_PREFIX="$PWD/../install"

export PKG_CONFIG_PATH="$TREXIO_PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$TREXIO_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export CPPFLAGS="-I$TREXIO_PREFIX/include ${CPPFLAGS:-}"
export LDFLAGS="-L$TREXIO_PREFIX/lib64 ${LDFLAGS:-}"

echo "TREXIO_PREFIX = $TREXIO_PREFIX"

echo "Checking TREXIO..."
pkg-config --modversion trexio
pkg-config --cflags trexio
pkg-config --libs trexio


echo "=========================================================="
echo " Phase 1: Generating QMCkl Source Files (Emacs/Autotools) "
echo "=========================================================="
# 1. Tangle the .org files into .c, .h, and .F90 files
ACLOCAL_PATH=/usr/share/aclocal ./autogen.sh
#./autogen.sh

# 2. Configure the environment to generate config.h and headers
# We disable compilation features because CMake will handle the actual build
./configure --disable-fortran --disable-python
#./configure --with-blas=openblas --with-lapack=openblas --disable-fortran --disable-python

# 3. built the library
make -j 4 


echo "=========================================================="
echo " Phase 2: Python Environment & TREXIO Setup               "
echo "=========================================================="
# If the test suite requires the Python venv, we initialize it here
if [ -f "tests/testvenv.sh" ]; then
    echo "Executing Python virtual environment setup..."
    cp files_qmckl_kokkos/testvenv_jureca.sh tests/
    chmod +x tests/testvenv_jureca.sh
    ./tests/testvenv_jureca.sh
fi

echo "=========================================================="
echo " SUCCESS: Source files generated. Ready for CMake builds. "
echo "=========================================================="
