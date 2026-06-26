cmake_minimum_required(VERSION 3.18)

# Globally define the NVIDIA Ampere architecture (RTX 3050)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Natively add CUDA as a first-class language
project(QMCkl_Kokkos C CXX Fortran CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Strictly enforce C++17 for NVCC 12.0 compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(HAVE_HPC "IVDEP=" HAVE_KOKKOS)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})



# ==========================================================
# If Kokkos enables CUDA, use NVCC for the Kokkos kernel file
# ==========================================================
if(Kokkos_ENABLE_CUDA)
  enable_language(CUDA)
  set_source_files_properties(src/qmckl_orbitals_kokkos.cpp PROPERTIES LANGUAGE CUDA COMPILE_OPTIONS "--extended-lambda")
endif()


# ==========================================================
#  KOKKOS & KOKKOS-KERNELS FETCHCONTENT INTEGRATION
# ==========================================================
include(FetchContent)

# Fetch Core Kokkos (Memory and Execution Spaces)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        4.4.01
)
FetchContent_MakeAvailable(Kokkos)

# Fetch Kokkos Kernels (for KokkosBlas::gemm)
FetchContent_Declare(
  KokkosKernels
  GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
  GIT_TAG        4.4.01
)
FetchContent_MakeAvailable(KokkosKernels)

# ==========================================================
#  Local Dependencies (TREXIO, BLAS, LAPACK)
# ==========================================================
find_path(TREXIO_INCLUDE_DIR trexio.h REQUIRED)
find_library(TREXIO_LIBRARY trexio REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

message(STATUS "Found TREXIO Include: ${TREXIO_INCLUDE_DIR}")
message(STATUS "Found TREXIO Library: ${TREXIO_LIBRARY}")

# ==========================================================
#  Gather Sources
# ==========================================================
file(GLOB QMCKL_C_SOURCES "src/*.c")
file(GLOB QMCKL_CXX_SOURCES "src/*.cpp") # Glob the upcoming Kokkos kernels
file(GLOB QMCKL_F_SOURCES "src/*.f90" "src/*.F90" "include/*.f90" "include/*.F90") 

list(FILTER QMCKL_C_SOURCES EXCLUDE REGEX ".*_(acc|omp)\\.c$")
list(FILTER QMCKL_F_SOURCES EXCLUDE REGEX ".*_fh.*")

#  Build the QMCkl shared library
add_library(qmckl SHARED ${QMCKL_C_SOURCES} ${QMCKL_CXX_SOURCES} ${QMCKL_F_SOURCES})


# Remove Kokkos targets from here. Only include paths belong here.
target_include_directories(qmckl PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${TREXIO_INCLUDE_DIR}>
)

# Add Kokkos::kokkoskernels to the linking stage
#  Link against TREXIO, Math, BLAS, LAPACK, and Kokkos
target_link_libraries(qmckl PUBLIC 
    ${TREXIO_LIBRARY} m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} 
    Kokkos::kokkos Kokkos::kokkoskernels
)

#  Ensure run_test also links the kernels
#  Build the Benchmark Executable
add_executable(run_test tests/main_test.c)

target_link_libraries(run_test qmckl Kokkos::kokkos Kokkos::kokkoskernels)
