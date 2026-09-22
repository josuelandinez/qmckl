

### I. Install Dependencies

1. Clone and install `trexio` on your target system.

### II. Build QMCkl (Kokkos Branch)

1. Clone the `qmckl` repository (targeting the Kokkos branch): `[https://github.com/josuelandinez/qmckl/tree/qmckl_kokkos](https://github.com/josuelandinez/qmckl/tree/qmckl_kokkos)`
2. Navigate into the cloned `qmckl` directory.
3. Modify and run `files_qmckl_kokkos/generate_sources_jureca.sh` to generate the required source files.
4. Copy the following files from `files_qmckl_kokkos/` directly into the `src/` directory:
* `qmckl_ao.c`
* `qmckl_mo.c`
* `qmckl_context.c`
* `qmckl_orbitals_kokkos.c`


5. Modify the build scripts to pass the correct include and linking paths for your environment.
6. Execute the appropriate script to build the library:
* **CPU:** Run `build_cpu_pdebug_minimal.sh`
* **GPU:** Run `build_gpu_pdebug_minimal.sh`



### III. Build QMCkl Benchmarks

1. Clone the benchmark repository: `[https://github.com/TREX-CoE/qmckl_bench](https://github.com/TREX-CoE/qmckl_bench)`
2. From your `qmckl` setup, copy `bench_aos.c` and `bench_mos.c` (located in `files_qmckl_kokkos/`) into the `qmckl_bench/src/` directory.
3. Copy `CMakeLists_bench.txt` and `build_qmckl_bench_minimal.sh` from `files_qmckl_kokkos/` into the root `qmckl_bench/` directory.
4. Navigate into the `qmckl_bench` directory.
5. Modify `build_qmckl_bench_minimal.sh` to match your loaded modules and the include/linking paths of your dependencies, then execute the script to compile the benchmarks.