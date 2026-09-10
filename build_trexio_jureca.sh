#!/bin/sh

module --force purge

module load Stages/2026 \
       GCC/14.3.0 \
       OpenMPI/5.0.8 \
       CUDA/13 \
       CMake/3.31.8  \
       Doxygen/1.14.0 \
       HDF5/1.14.6-serial

# 2. Clone the specific TREXIO release
#git clone --branch v2.4.1 https://github.com/TREX-CoE/trexio.git
git clone https://github.com/TREX-CoE/trexio.git
cd trexio

# 3. Configure with CMake
# We force HDF5 ON and set the install prefix to match the QMCkl script
mkdir build_cmake
cd build_cmake
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install -DENABLE_HDF5=ON

# 4. Compile and Install
make -j 4
make install

# 5. Verify trexio installation
ls $PWD/../../install/include
ls $PWD/../../install/lib64

cd ..
