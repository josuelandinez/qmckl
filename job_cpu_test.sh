#!/bin/bash
#SBATCH --job-name="cpu-build-test"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=dc-cpu-devel
#SBATCH --cpus-per-task=128
#SBATCH --account=zam
#SBATCH --output=cpu_build_test_%j.out.txt
#SBATCH --error=cpu_build_test_%j.err.txt
#SBATCH --disable-turbomode

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


export BIN_DIR=${PWD}/build_cpu
export OMPI_MCA_pml=ucx

export OMP_NUM_THREADS=128
export OMP_PROC_BIND=True
export OMP_PLACES=cores
export UCX_LOG_LEVEL=fatal

cd ${BIN_DIR}

TREXIO_PREFIX=$PWD/../../install


echo "=========================================================="
echo " Executing Kokkos GPU Validation                          "
echo "=========================================================="
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib64:$LD_LIBRARY_PATH    


# We link dynamically against the built library using the correct relative paths                                                                                                                                                                        
gcc -I../include -I../src -I$TREXIO_PREFIX/include ../tests/verify_orbitals.c \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -o verify_orbitals_cpu

###source ../tests/testvenv/bin/activate                                                                                                                                                                                                                
export LD_LIBRARY_PATH=$PWD:build_cpu:$LD_LIBRARY_PATH

ldd ./verify_orbitals_cpu | grep -E 'omp|kokkos|qmckl|trexio'

which nvcc
nvcc --version

./verify_orbitals_cpu ../files_qmckl_kokkos/water.hdf5



#adding the other tests

gfortran -I./modules -c ../tests/test_qmckl_ao_f.F90 -o test_qmckl_ao_f.o

gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_ao.c test_qmckl_ao_f.o \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -lgfortran -o test_qmckl_ao_cpu

./test_qmckl_ao_cpu

gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_mo.c \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -o test_qmckl_mo_cpu

./test_qmckl_mo_cpu
