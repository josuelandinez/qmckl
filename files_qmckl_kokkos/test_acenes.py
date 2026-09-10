import os
import math
import numpy as np
import trexio
from pyscf import gto, scf
from pyscf.tools import trexio as pyscf_trexio

def generate_acene_xyz(n_rings, r_cc=1.40, r_ch=1.08):
    """
    Generate planar D2h coordinates for an n-acene (C_{4n+2} H_{2n+4}).
    Rings are tiled along the X axis, centered around X=0.
    """
    dx = math.sqrt(3.0) * r_cc
    atoms = []
    
    # Center the entire chain at x = 0
    x_offset = -0.5 * (n_rings - 1) * dx
    
    for i in range(n_rings):
        xc = x_offset + i * dx
        
        # Carbon atoms (top and bottom backbone)
        atoms.append(('C', (xc - 0.25 * dx,  0.5 * r_cc, 0.0)))
        atoms.append(('C', (xc + 0.25 * dx,  0.5 * r_cc, 0.0)))
        atoms.append(('C', (xc - 0.25 * dx, -0.5 * r_cc, 0.0)))
        atoms.append(('C', (xc + 0.25 * dx, -0.5 * r_cc, 0.0)))
        
        # Top and bottom terminal H atoms for each ring
        atoms.append(('H', (xc - 0.25 * dx,  0.5 * r_cc + r_ch, 0.0)))
        atoms.append(('H', (xc + 0.25 * dx,  0.5 * r_cc + r_ch, 0.0)))
        atoms.append(('H', (xc - 0.25 * dx, -0.5 * r_cc - r_ch, 0.0)))
        atoms.append(('H', (xc + 0.25 * dx, -0.5 * r_cc - r_ch, 0.0)))
        
        # Leftmost ring: add outer bridgehead carbons and lateral hydrogens
        if i == 0:
            x_left = xc - 0.5 * dx
            atoms.append(('C', (x_left, 0.0, 0.0)))
            atoms.append(('H', (x_left - r_ch, 0.0, 0.0)))
            
        # Rightmost ring: add outer bridgehead carbons and lateral hydrogens
        if i == n_rings - 1:
            x_right = xc + 0.5 * dx
            atoms.append(('C', (x_right, 0.0, 0.0)))
            atoms.append(('H', (x_right + r_ch, 0.0, 0.0)))

    # Deduplicate internal ring edges where atoms might be shared/close
    unique_atoms = []
    for sym, coord in atoms:
        is_dup = False
        for _, ucoord in unique_atoms:
            if math.dist(coord, ucoord) < 0.2:
                is_dup = True
                break
        if not is_dup:
            unique_atoms.append((sym, coord))
            
    return unique_atoms

def write_xyz(filename, atoms, comment=""):
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for sym, (x, y, z) in atoms:
            f.write(f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")

def export_acene_trexio(atoms, h5_filename, basis_name='cc-pvdz'):
    # Build geometry string for PySCF
    atom_str = "; ".join([f"{s} {c[0]} {c[1]} {c[2]}" for s, c in atoms])
    
    mol = gto.M(
        atom=atom_str,
        basis=basis_name,
        cart=False,
        symmetry=False
    )
    
    print(f"--> Computing RHF for {h5_filename} | Basis: {basis_name} | AOs: {mol.nao_nr()}...")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    mf.kernel()
    
    if os.path.exists(h5_filename):
        os.remove(h5_filename)
        
    pyscf_trexio.to_trexio(mf, h5_filename)
    
    # Inject QMCkL strict data
    with trexio.File(h5_filename, 'u', trexio.TREXIO_HDF5) as f:
        n_up, n_dn = mol.nelec
        if not trexio.has_electron_up_num(f):
            trexio.write_electron_up_num(f, n_up)
        if not trexio.has_electron_dn_num(f):
            trexio.write_electron_dn_num(f, n_dn)
        if not trexio.has_state_id(f):
            trexio.write_state_id(f, 0)
            
        int64_num = trexio.get_int64_num(f)
        det_list = [
            list(trexio.to_bitfield_list(int64_num, list(range(n_up)))) +
            list(trexio.to_bitfield_list(int64_num, list(range(n_dn))))
        ]
        trexio.write_determinant_list(f, 0, 1, det_list)
        trexio.write_determinant_coefficient(f, 0, 1, [1.0])
        
        if not trexio.has_nucleus_repulsion(f):
            trexio.write_nucleus_repulsion(f, float(mf.energy_nuc()))
            
        num_ao = trexio.read_ao_num(f)
        num_shells = trexio.read_basis_shell_num(f)
        num_prims = trexio.read_basis_prim_num(f)
        
        if not trexio.has_ao_normalization(f):
            trexio.write_ao_normalization(f, [1.0] * num_ao)
        if not trexio.has_basis_shell_factor(f):
            trexio.write_basis_shell_factor(f, [1.0] * num_shells)
        if not trexio.has_basis_prim_factor(f):
            trexio.write_basis_prim_factor(f, [1.0] * num_prims)

# --- Execution Suite ---
if __name__ == "__main__":
    os.makedirs("acene_geometries", exist_ok=True)
    os.makedirs("acene_trexio_db", exist_ok=True)
    
    acene_names = {
        1: "benzene",
        2: "naphthalene",
        3: "anthracene",
        4: "tetracene",
        5: "pentacene",
        6: "hexacene"
    }
    
    # Define the basis sets you want to benchmark
    basis_sets = ["sto-3g", "cc-pvdz", "aug-cc-pvdz"]
    #basis_sets = ["sto-3g"]
    
    for n_rings, name in acene_names.items():
        atoms = generate_acene_xyz(n_rings)
        xyz_path = f"acene_geometries/{name}.xyz"
        write_xyz(xyz_path, atoms, comment=f"{name} (n={n_rings})")
        print(f"Saved: {xyz_path} ({len(atoms)} atoms)")
        
        for basis in basis_sets:
            h5_path = f"acene_trexio_db/{name}_{basis}.h5"
            export_acene_trexio(atoms, h5_path, basis_name=basis)

    print("\nDatabase creation complete. All .h5 files are ready for Kokkos kernels.")
