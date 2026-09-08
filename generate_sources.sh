#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "=========================================================="
echo " Phase 1: Generating QMCkl Source Files (Emacs/Autotools) "
echo "=========================================================="
# 1. Tangle the .org files into .c, .h, and .F90 files
./autogen.sh

# 2. Configure the environment to generate config.h and headers
# We disable compilation features because CMake will handle the actual build
./configure --disable-fortran --disable-python

# 3. built the library
make -j 4 


echo "=========================================================="
echo " Phase 2: Python Environment & TREXIO Setup               "
echo "=========================================================="
# If the test suite requires the Python venv, we initialize it here
if [ -f "tests/testvenv.sh" ]; then
    echo "Executing Python virtual environment setup..."
    cp files_qmckl_kokkos/testvenv.sh tests/
    chmod +x tests/testvenv.sh
    ./tests/testvenv.sh
fi

echo "=========================================================="
echo " SUCCESS: Source files generated. Ready for CMake builds. "
echo "=========================================================="
