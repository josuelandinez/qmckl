#!/bin/bash -x
#SBATCH --job-name="acenes_gen"
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=acenes-%j.out
#SBATCH --error=acenes-%j.err
#SBATCH --partition=dc-cpu
#SBATCH --gres=NONE
#SBATCH --account=zam
#SBATCH --mem=512G
#SBATCH --disable-turbomode

module --force purge

module load Stages/2026 \
       GCC/14.3.0 \
       OpenBLAS/0.3.30 \
       virtualenv/20.32.0 \
       HDF5/1.14.6-serial

# Set TREXIO prefix path before using it in compiler/linker flags
TREXIO_PREFIX=$PWD/../install

export CFLAGS="-I$EBROOTHDF5/include -I$TREXIO_PREFIX/include"
export LDFLAGS="-L$EBROOTHDF5/lib -L$TREXIO_PREFIX/lib"
export LD_LIBRARY_PATH="$EBROOTHDF5/lib:$TREXIO_PREFIX/lib:$LD_LIBRARY_PATH"

# Thread control: tell OpenBLAS, PySCF, and OpenMP how many cores to use
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-128}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Pin threads to physical cores across the 2 sockets
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Activate virtual environment
source tests/testvenv/bin/activate

# Execute generator
python test_acenes.py
