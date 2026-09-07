#!/bin/bash

TREXIO_PREFIX=$HOME/Codes/QMCkl_Kokkos/install

# Initialize and activate the isolated environment
python3 -m venv tests/testvenv
source tests/testvenv/bin/activate

# Map the C compiler and linker to your local, HDF5-enabled TREXIO library
export CFLAGS="-I$TREXIO_PREFIX/include"
export LDFLAGS="-L$TREXIO_PREFIX/lib"
export LD_LIBRARY_PATH="$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH"

# Upgrade pip to ensure the local compilation engine is up to date
pip install --upgrade pip

# Force source compilation of TREXIO to inherit local HDF5 capabilities
pip install --no-binary trexio trexio

# Install the remaining analytical tools
pip install pyscf trexio-tools h5py
