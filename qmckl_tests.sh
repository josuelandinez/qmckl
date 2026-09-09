#!/bin/bash
set -e 

TREXIO_PREFIX=$HOME/Codes/QMCkl_Kokkos/install

## cpu test

cd build_cpu


export TREXIO_PREFIX=$HOME/Codes/QMCkl_Kokkos/install
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH


# Compile the Fortran verification math, including the modules path
gfortran -I./modules -c ../tests/test_qmckl_ao_f.F90 -o test_qmckl_ao_f.o

# Compile the C test and link
gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_ao.c test_qmckl_ao_f.o \
    -L. -L$TREXIO_PREFIX/lib \
    -lqmckl -ltrexio -lm -lstdc++ -lgfortran -o test_qmckl_ao_cpu

# Execute
./test_qmckl_ao_cpu


# Compile the pure C test and link against your Kokkos library
gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_mo.c \
    -L. -L$TREXIO_PREFIX/lib \
    -lqmckl -ltrexio -lm -lstdc++ -o test_qmckl_mo_cpu

# Execute the specific C test
./test_qmckl_mo_cpu


## cpu test


cd ../build_gpu



export TREXIO_PREFIX=$HOME/Codes/QMCkl_Kokkos/install
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH

# Compile the Fortran verification math, including the modules path
gfortran -I./modules -c ../tests/test_qmckl_ao_f.F90 -o test_qmckl_ao_f.o

# Compile the C test and link
gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_ao.c test_qmckl_ao_f.o \
    -L. -L$TREXIO_PREFIX/lib \
    -lqmckl -ltrexio -lm -lstdc++ -lgfortran -o test_qmckl_ao_gpu

# Execute
./test_qmckl_ao_gpu


# Compile the pure C test and link against your Kokkos library
gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_mo.c \
    -L. -L$TREXIO_PREFIX/lib \
    -lqmckl -ltrexio -lm -lstdc++ -o test_qmckl_mo_gpu

# Execute the specific C test
./test_qmckl_mo_gpu


