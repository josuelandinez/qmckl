#!/bin/bash
#SBATCH --job-name="gpu-build-test"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=dc-gpu-devel
#SBATCH --cpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --account=zam
#SBATCH --output=gpu_build_test_%j.out.txt
#SBATCH --error=gpu_build_test_%j.err.txt
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


export BIN_DIR=${PWD}/build_gpu
export OMPI_MCA_pml=ucx

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=True
export OMP_PLACES=cores
export UCX_LOG_LEVEL=fatal

cd ${BIN_DIR}

TREXIO_PREFIX=$PWD/../../install

echo $CUDA_VISIBLE_DEVICES

nvidia-smi --query-gpu=name,compute_cap --format=csv




echo "=========================================================="
echo " Executing Kokkos GPU Validation                          "
echo "=========================================================="
export LD_LIBRARY_PATH=$PWD:$TREXIO_PREFIX/lib64:$LD_LIBRARY_PATH    


# We link dynamically against the built library using the correct relative paths                                                                                                                                                                        
gcc -I../include -I../src -I$TREXIO_PREFIX/include ../tests/verify_orbitals.c \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -o verify_orbitals_gpu

###source ../tests/testvenv/bin/activate                                                                                                                                                                                                                
export LD_LIBRARY_PATH=$PWD:build_gpu:$LD_LIBRARY_PATH

ldd ./verify_orbitals_gpu | grep -E 'cuda|cudart|kokkos|qmckl|trexio'

which nvcc
nvcc --version

./verify_orbitals_gpu ../files_qmckl_kokkos/water.hdf5



#adding the other tests

gfortran -I./modules -c ../tests/test_qmckl_ao_f.F90 -o test_qmckl_ao_f.o

gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_ao.c test_qmckl_ao_f.o \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -lgfortran -o test_qmckl_ao_gpu

./test_qmckl_ao_gpu

gcc -I../include -I../src -I$TREXIO_PREFIX/include \
    ../tests/test_qmckl_mo.c \
    -L. -L$TREXIO_PREFIX/lib64 \
    -lqmckl -ltrexio -lm -lstdc++ -o test_qmckl_mo_gpu

./test_qmckl_mo_gpu
