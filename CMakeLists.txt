cmake_minimum_required(VERSION 3.18)

# We dont enfoncer the arquitecture here it will pass trough the build script
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(QMCkl_Kokkos C Fortran) # CXX and CUDA are deferred

# ==========================================================
# Native Sources & Dependencies
# ==========================================================
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

find_path(TREXIO_INCLUDE_DIR trexio.h REQUIRED)
find_library(TREXIO_LIBRARY trexio REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

file(GLOB QMCKL_C_SOURCES "src/*.c")
file(GLOB QMCKL_F_SOURCES "src/*.f90" "src/*.F90" "include/*.f90" "include/*.F90") 
list(FILTER QMCKL_C_SOURCES EXCLUDE REGEX ".*_(acc|omp)\\.c$")
list(FILTER QMCKL_F_SOURCES EXCLUDE REGEX ".*_fh.*")

# Initialize the core library variables
set(QMCKL_ALL_SOURCES ${QMCKL_C_SOURCES} ${QMCKL_F_SOURCES})
set(QMCKL_ALL_LINK_LIBS ${TREXIO_LIBRARY} m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# ==========================================================
#  Kokkos Acceleration Toggle
# ==========================================================
option(ENABLE_KOKKOS "Enable Kokkos GPU/CPU acceleration" ON)

if(ENABLE_KOKKOS)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    add_compile_definitions(HAVE_HPC "IVDEP=" HAVE_KOKKOS HAVE_TREXIO)
    
    if(Kokkos_ENABLE_CUDA)
        enable_language(CUDA)
        set_source_files_properties(src/qmckl_orbitals_kokkos.cpp PROPERTIES LANGUAGE CUDA COMPILE_OPTIONS "--extended-lambda")
    endif()

    # Fetch Dependencies ONLY if Kokkos is enabled
    include(FetchContent)
    FetchContent_Declare(Kokkos GIT_REPOSITORY https://github.com/kokkos/kokkos.git GIT_TAG master)
    FetchContent_MakeAvailable(Kokkos)
    FetchContent_Declare(KokkosKernels GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git GIT_TAG master)
    FetchContent_MakeAvailable(KokkosKernels)

    # Append CXX sources and Linker libraries
    file(GLOB QMCKL_CXX_SOURCES "src/*.cpp")
    list(APPEND QMCKL_ALL_SOURCES ${QMCKL_CXX_SOURCES})
    list(APPEND QMCKL_ALL_LINK_LIBS Kokkos::kokkos Kokkos::kokkoskernels)
else()
    add_compile_definitions(HAVE_HPC "IVDEP=" HAVE_TREXIO)
endif()

# ==========================================================
#  Library and Executable Generation
# ==========================================================
add_library(qmckl SHARED ${QMCKL_ALL_SOURCES})

target_include_directories(qmckl PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${TREXIO_INCLUDE_DIR}>
)

target_link_libraries(qmckl PUBLIC ${QMCKL_ALL_LINK_LIBS})

# Build the Benchmark Executables
add_executable(run_test tests/main_test.c)
target_link_libraries(run_test qmckl ${QMCKL_ALL_LINK_LIBS})

add_executable(verify_orbitals tests/verify_orbitals.c)
target_link_libraries(verify_orbitals qmckl ${QMCKL_ALL_LINK_LIBS})
