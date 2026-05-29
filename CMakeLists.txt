cmake_minimum_required(VERSION 3.16)
project(QMCkl_Base C Fortran)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_definitions(HAVE_HPC "IVDEP=")

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

# 1. Find locally installed TREXIO
find_path(TREXIO_INCLUDE_DIR trexio.h REQUIRED)
find_library(TREXIO_LIBRARY trexio REQUIRED)

# ==========================================================
# MINIMAL FIX: Find BLAS and LAPACK
# ==========================================================
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

message(STATUS "Found TREXIO Include: ${TREXIO_INCLUDE_DIR}")
message(STATUS "Found TREXIO Library: ${TREXIO_LIBRARY}")

# 2. Gather Sources
file(GLOB QMCKL_C_SOURCES "src/*.c")
file(GLOB QMCKL_F_SOURCES "src/*.f90" "src/*.F90" "include/*.f90" "include/*.F90") 

list(FILTER QMCKL_C_SOURCES EXCLUDE REGEX ".*_(acc|omp)\\.c$")
list(FILTER QMCKL_F_SOURCES EXCLUDE REGEX ".*_fh.*")

# 3. Build the QMCkl shared library
add_library(qmckl SHARED ${QMCKL_C_SOURCES} ${QMCKL_F_SOURCES})

# 4. Include directories
target_include_directories(qmckl PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${TREXIO_INCLUDE_DIR}>
)

# 5. Link against TREXIO, Math, BLAS, and LAPACK
# ==========================================================
# MINIMAL FIX: Add ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
# ==========================================================
target_link_libraries(qmckl PUBLIC ${TREXIO_LIBRARY} m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# 6. Build the Benchmark Executable
add_executable(run_test tests/main_test.c)
target_link_libraries(run_test qmckl)
