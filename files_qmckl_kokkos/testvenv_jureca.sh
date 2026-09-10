#!/bin/bash

set -e

module --force purge

module load Stages/2026 \
       GCC/14.3.0 \
       OpenBLAS/0.3.30 \
       virtualenv/20.32.0 \
       HDF5/1.14.6-serial
       


TREXIO_PREFIX=$PWD/../install

# Initialize and activate the isolated environment
python3 -m venv tests/testvenv
source tests/testvenv/bin/activate

# Map the C compiler and linker to your local, HDF5-enabled TREXIO library

export CFLAGS="-I$EBROOTHDF5/include -I$TREXIO_PREFIX/include"
export LDFLAGS="-L$EBROOTHDF5/lib -L$TREXIO_PREFIX/lib"
export LD_LIBRARY_PATH="$EBROOTHDF5/lib:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH"

# Upgrade pip to ensure the local compilation engine is up to date
pip install --upgrade pip

# Force source compilation of TREXIO to inherit local HDF5 capabilities
#pip install --no-binary trexio trexio

# Install the remaining analytical tools
pip install pyscf  pyscf-forge trexio-tools h5py 
