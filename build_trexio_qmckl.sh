#!/bin/sh

# 1. Create your workspace
mkdir -p $HOME/Codes/QMCkl_Kokkos
cd $HOME/Codes/QMCkl_Kokkos
echo ${PWD}
sleep 2

# 2. Clone the specific TREXIO release
#git clone --branch v2.4.1 https://github.com/TREX-CoE/trexio.git
git clone https://github.com/TREX-CoE/trexio.git
cd trexio

# 3. Configure with CMake
# We force HDF5 ON and set the install prefix to match your QMCkl script
mkdir build_cmake
cd build_cmake
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/Codes/QMCkl_Kokkos/install -DENABLE_HDF5=ON

# 4. Compile and Install
make -j 4
make install

# 5. Verify trexio installation
ls $HOME/Codes/QMCkl_Kokkos/install/include
ls $HOME/Codes/QMCkl_Kokkos/install/lib

# 6. BUild QMCkl Kokkos
cd ..
echo ${PWD}
sleep 5
git clone git@github.com:josuelandinez/qmckl.git

cd $HOME/Codes/QMCkl_Kokkos/qmckl
git checkout qmckl_kokkos
chmod +x build_cpu.sh
./build_cpu.sh
