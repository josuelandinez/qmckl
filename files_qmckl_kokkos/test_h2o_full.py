import os
import trexio
from pyscf import gto, scf
from pyscf.tools import trexio as pyscf_trexio

filename = 'water.hdf5'

if os.path.exists(filename):
    os.remove(filename)

# 1. Run Hartree-Fock
mol = gto.M(
    atom = 'O 0 0 0; H 0 1 0; H 0 0 1',
    basis = 'sto-3g',
    cart = False   # Keep False for spherical harmonics; True if Kokkos kernels expect Cartesian (6d, 10f)
)
mf = scf.RHF(mol)
mf.kernel()

# 2. Export basic quantum chemistry structure
pyscf_trexio.to_trexio(mf, filename)

# 3. Patch missing QMCkL groups (electrons, determinants, normalizations)
with trexio.File(filename, 'u', trexio.TREXIO_HDF5) as f:
    n_up, n_dn = mol.nelec

    # --- Electron count ---
    if not trexio.has_electron_up_num(f):
        trexio.write_electron_up_num(f, n_up)
    if not trexio.has_electron_dn_num(f):
        trexio.write_electron_dn_num(f, n_dn)

    # --- State & Determinant initialization (Single RHF Slater Determinant) ---
    if not trexio.has_state_id(f):
        trexio.write_state_id(f, 0)

    # Determinant bitfields for RHF ground state
    int64_num = trexio.get_int64_num(f)
    det_list = [
        list(trexio.to_bitfield_list(int64_num, list(range(n_up)))) +
        list(trexio.to_bitfield_list(int64_num, list(range(n_dn))))
    ]
    trexio.write_determinant_list(f, 0, 1, det_list)
    trexio.write_determinant_coefficient(f, 0, 1, [1.0])

    # --- Classical nuclear repulsion ---
    if not trexio.has_nucleus_repulsion(f):
        trexio.write_nucleus_repulsion(f, float(mf.energy_nuc()))

    # --- Normalization & Shell Scaling Factors for QMCkL AO Kernels ---
    num_ao = trexio.read_ao_num(f)
    num_shells = trexio.read_basis_shell_num(f)
    num_prims = trexio.read_basis_prim_num(f)

    if not trexio.has_ao_normalization(f):
        trexio.write_ao_normalization(f, [1.0] * num_ao)
    if not trexio.has_basis_shell_factor(f):
        trexio.write_basis_shell_factor(f, [1.0] * num_shells)
    if not trexio.has_basis_prim_factor(f):
        trexio.write_basis_prim_factor(f, [1.0] * num_prims)

print("Generated and fully patched water.hdf5 for QMCkL.")
