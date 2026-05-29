## QMCkl HPC Architecture Dependencies

To compile the QMCkl HPC baseline, the build system relies on a two-phase compilation: Emacs (to parse `.org` files into source code) and CMake (to compile the mixed C/Fortran libraries).

### Local Installation (Ubuntu/Debian)
Ensure you have the GNU compiler suite, Emacs, CMake, and the required math/data backends:
\`\`\`bash
sudo apt update
sudo apt install -y build-essential gfortran emacs cmake autoconf automake libtool pkg-config
sudo apt install -y libopenblas-dev liblapack-dev libhdf5-dev
\`\`\`

### HPC Cluster Environment (Modules)
When compiling on a supercomputer, do **not** use the system default compilers. Load the optimized toolchains provided by your cluster administrators. 

A standard module load sequence looks like this:
\`\`\`bash
# 1. Load the core compiler suite (GCC or Intel)
module load gcc/11.2.0  # Or higher

# 2. Load the build tools
module load cmake/3.24
module load emacs/28.2

# 3. Load the optimized Math and Data engines
module load openblas/0.3.21
module load hdf5/1.12.2
\`\`\`
*(Check your specific cluster with `module avail` to find the exact version numbers).*

### Building the Baseline
Once TREXIO is compiled and installed locally, set its location and run the automated script:
\`\`\`bash
export TREXIO_PREFIX=/path/to/your/trexio/install
./build_baseline.sh
\`\`\`