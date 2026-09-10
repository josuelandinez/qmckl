from pyscf import gto, scf
# Import the native PySCF-to-TREXIO bridge
from pyscf.tools import trexio as pyscf_trexio

# Create a Water Molecule with sto-3g
mol = gto.M(
    atom = 'O 0 0 0; H 0 1 0; H 0 0 1',
    basis = 'sto-3g'
)

# Run Hartree-Fock
mf = scf.RHF(mol)
mf.kernel()

# Directly export the SCF object to a TREXIO HDF5 file
pyscf_trexio.to_trexio(mf, 'water.hdf5')

print("Generated water.hdf5 successfully.")
