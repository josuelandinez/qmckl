import os
import shutil
from pyscf import gto, scf
import trexio
from trexio_tools.converters import runner

filename = 'water.trexio'
if os.path.exists(filename):
    shutil.rmtree(filename) if os.path.isdir(filename) else os.remove(filename)

mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto-3g')
mf = scf.RHF(mol).run()

# trexio_tools runner handles the full PySCF object conversion directly:
runner.run(
    driver="pyscf",
    input_data=mf,
    output_file=filename,
    back_end="text"   # or "hdf5" if HDF5 backend is active
)
