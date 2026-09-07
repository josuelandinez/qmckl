import numpy as np
import trexio
from pyscf import gto, scf
from pyscf.tools import trexio as pyscf_trexio

filename = 'water.trexio'

print("=== Running PySCF ===")
mol = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'sto-3g')
mf = scf.RHF(mol)
mf.kernel()

print(f"\n=== Exporting to {filename} (TEXT Backend) ===")
pyscf_trexio.to_trexio(mf, filename)

print("\n=== Patching QMC Strict Data (ASCII) ===")
try:
    with trexio.File(filename, 'u', trexio.TREXIO_TEXT) as f:
        n_up, n_dn = mol.nelec
        if not trexio.has_electron(f):
            trexio.write_electron_up_num(f, n_up)
            trexio.write_electron_dn_num(f, n_dn)
        if not trexio.has_state(f):
            trexio.write_state_id(f, 0)
        if not trexio.has_determinant(f):
            int64_num = trexio.get_int64_num(f)
            det_list = [list(trexio.to_bitfield_list(int64_num, list(range(n_up)))) + list(trexio.to_bitfield_list(int64_num, list(range(n_dn))))]
            trexio.write_determinant_list(f, 0, 1, det_list)
            trexio.write_determinant_coefficient(f, 0, 1, [1.0])
        
        if not trexio.has_nucleus_repulsion(f): trexio.write_nucleus_repulsion(f, 0.0)
        if not trexio.has_ao_normalization(f): trexio.write_ao_normalization(f, [1.0] * trexio.read_ao_num(f))
        if not trexio.has_basis_shell_factor(f): trexio.write_basis_shell_factor(f, [1.0] * trexio.read_basis_shell_num(f))
        if not trexio.has_basis_prim_factor(f): trexio.write_basis_prim_factor(f, [1.0] * trexio.read_basis_prim_num(f))
        print("- QMC Groups and explicit scalars injected successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
