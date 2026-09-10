#!/usr/bin/env python3
import os
import math
import shutil
import numpy as np
import trexio
from pyscf import gto, scf
from pyscf.tools import trexio as pyscf_trexio

def generate_acene_geometry(n_rings: int, r_cc: float = 1.40, r_ch: float = 1.08):
    """
    Constructs planar D2h coordinates for an n-acene (C_{4n+2} H_{2n+4}).
    Rings are tiled along the X-axis, centered at X=0.
    """
    dx = math.sqrt(3.0) * r_cc
    raw_atoms = []
    
    # Center the entire chain along the X-axis
    x_offset = -0.5 * (n_rings - 1) * dx
    
    for i in range(n_rings):
        xc = x_offset + i * dx
        
        # Carbon backbone (top and bottom of each ring)
        raw_atoms.append(('C', (xc - 0.25 * dx,  0.5 * r_cc, 0.0)))
        raw_atoms.append(('C', (xc + 0.25 * dx,  0.5 * r_cc, 0.0)))
        raw_atoms.append(('C', (xc - 0.25 * dx, -0.5 * r_cc, 0.0)))
        raw_atoms.append(('C', (xc + 0.25 * dx, -0.5 * r_cc, 0.0)))
        
        # Outer C-H bonds (top and bottom)
        raw_atoms.append(('H', (xc - 0.25 * dx,  0.5 * r_cc + r_ch, 0.0)))
        raw_atoms.append(('H', (xc + 0.25 * dx,  0.5 * r_cc + r_ch, 0.0)))
        raw_atoms.append(('H', (xc - 0.25 * dx, -0.5 * r_cc - r_ch, 0.0)))
        raw_atoms.append(('H', (xc + 0.25 * dx, -0.5 * r_cc - r_ch, 0.0)))
        
        # Lateral boundary terminations
        if i == 0:
            x_left = xc - 0.5 * dx
            raw_atoms.append(('C', (x_left, 0.0, 0.0)))
            raw_atoms.append(('H', (x_left - r_ch, 0.0, 0.0)))
            
        if i == n_rings - 1:
            x_right = xc + 0.5 * dx
            raw_atoms.append(('C', (x_right, 0.0, 0.0)))
            raw_atoms.append(('H', (x_right + r_ch, 0.0, 0.0)))

    # Deduplicate shared junction boundaries
    unique_atoms = []
    for sym, coord in raw_atoms:
        if not any(math.dist(coord, ucoord) < 0.2 for _, ucoord in unique_atoms):
            unique_atoms.append((sym, coord))
            
    # Sort: Carbons first, then Hydrogens
    unique_atoms.sort(key=lambda item: (item[0] != 'C', item[1][0]))
    return unique_atoms

def write_xyz(filename: str, atoms: list, comment: str = ""):
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for sym, (x, y, z) in atoms:
            f.write(f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")

def run_scf_and_export_trexio(atoms: list, h5_filepath: str, basis_name: str):
    """
    Executes RHF in PySCF, exports via pyscf.tools.trexio,
    and patches internal QMCkL structures.
    """
    if os.path.exists(h5_filepath):
        os.remove(h5_filepath)
        
    atom_pyscf = "; ".join([f"{s} {c[0]} {c[1]} {c[2]}" for s, c in atoms])
    
    mol = gto.M(
        atom=atom_pyscf,
        basis=basis_name,
        cart=False,       # Spherical harmonics (standard in QMCkL)
        symmetry=False,
        verbose=0
    )
    
    n_elec = mol.nelectron
    n_ao = mol.nao_nr()
    print(f"  [PySCF] SCF Started -> Basis: {basis_name:<11s} | Elecs: {n_elec:3d} | AOs: {n_ao:4d}")
    
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    mf.max_cycle = 150
    mf.kernel()
    
    if not mf.converged:
        print(f"  [WARNING] SCF did not fully converge for {h5_filepath}!")

    # 1. Base export
    pyscf_trexio.to_trexio(mf, h5_filepath)
    
    # 2. QMCkL strict data patch
    with trexio.File(h5_filepath, 'u', trexio.TREXIO_HDF5) as f:
        n_up, n_dn = mol.nelec
        
        # Electron counts
        if not trexio.has_electron_up_num(f):
            trexio.write_electron_up_num(f, n_up)
        if not trexio.has_electron_dn_num(f):
            trexio.write_electron_dn_num(f, n_dn)
            
        # Ground state single determinant configuration
        if not trexio.has_state_id(f):
            trexio.write_state_id(f, 0)
            
        int64_num = trexio.get_int64_num(f)
        det_list = [
            list(trexio.to_bitfield_list(int64_num, list(range(n_up)))) +
            list(trexio.to_bitfield_list(int64_num, list(range(n_dn))))
        ]
        trexio.write_determinant_list(f, 0, 1, det_list)
        trexio.write_determinant_coefficient(f, 0, 1, [1.0])
        
        # Scalar nuclear repulsion energy
        if not trexio.has_nucleus_repulsion(f):
            trexio.write_nucleus_repulsion(f, float(mf.energy_nuc()))
            
        # Normalization and shell scaling factors
        total_aos = trexio.read_ao_num(f)
        total_shells = trexio.read_basis_shell_num(f)
        total_prims = trexio.read_basis_prim_num(f)
        
        if not trexio.has_ao_normalization(f):
            trexio.write_ao_normalization(f, [1.0] * total_aos)
        if not trexio.has_basis_shell_factor(f):
            trexio.write_basis_shell_factor(f, [1.0] * total_shells)
        if not trexio.has_basis_prim_factor(f):
            trexio.write_basis_prim_factor(f, [1.0] * total_prims)

    print(f"  [Output] Wrote {h5_filepath}")

def main():
    geom_dir = "benchmark_geometries"
    h5_size_dir = "benchmark_h5_scaling_size"
    h5_basis_dir = "benchmark_h5_scaling_basis"
    
    os.makedirs(geom_dir, exist_ok=True)
    os.makedirs(h5_size_dir, exist_ok=True)
    os.makedirs(h5_basis_dir, exist_ok=True)
    
    acene_names = {
        1: "benzene",
        2: "naphthalene",
        3: "anthracene",
        4: "tetracene",
        5: "pentacene",
        6: "hexacene"
    }

    print("=" * 68)
    print("AXIS 1: SYSTEM SIZE SCALING (Fixed Basis: cc-pvdz)")
    print("=" * 68)
    fixed_basis = "cc-pvdz"
    for n_rings, name in acene_names.items():
        atoms = generate_acene_geometry(n_rings)
        xyz_file = os.path.join(geom_dir, f"{name}.xyz")
        write_xyz(xyz_file, atoms, comment=f"{name} (n={n_rings}) D2h")
        
        h5_out = os.path.join(h5_size_dir, f"{name}_{fixed_basis}.h5")
        run_scf_and_export_trexio(atoms, h5_out, basis_name=fixed_basis)

    print("\n" + "=" * 68)
    print("AXIS 2: BASIS SET QUALITY SCALING (Fixed Molecule: tetracene)")
    print("=" * 68)
    tetracene_atoms = generate_acene_geometry(4)
    tetracene_basis_sets = [
        "cc-pvdz",
        "aug-cc-pvdz",
        "cc-pvtz",
        "aug-cc-pvtz",
        "cc-pvqz"
    ]
    
    for basis in tetracene_basis_sets:
        h5_out = os.path.join(h5_basis_dir, f"tetracene_{basis}.h5")
        run_scf_and_export_trexio(tetracene_atoms, h5_out, basis_name=basis)

    print("\nBenchmark test files successfully constructed.")

if __name__ == "__main__":
    main()
