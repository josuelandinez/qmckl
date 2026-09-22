#!/usr/bin/env python3
"""
Generates benzene at cc-pV5Z, kept as a separate, standalone script rather
than folded into test_benchmarks.py's two-axis (size / basis-quality) suite.

Purpose: cc-pV5Z is the first standard Dunning basis to introduce h-functions
(angular momentum L=5) for first-row atoms such as carbon. QMCkl's Kokkos AO
kernels currently support L up to 5 (s,p,d,f,g,h) via template instantiation;
L=6 (i-functions, introduced starting at cc-pV6Z) is outside that range and
returns QMCKL_FAILURE rather than being silently computed. This generates the
smallest reasonable molecule (benzene) at exactly the basis that reaches the
top of the currently-supported range, so a benchmark run against it verifies
the L=5 code path is genuinely exercised and correct - unlike the datasets
used throughout this project so far, whose actual max angular momentum was
never explicitly confirmed to reach L=5.

This is deliberately NOT a test of the L=6 failure path itself (that would
need cc-pV6Z or heavier elements) - it verifies the boundary that IS
supported actually works, which is a prerequisite for meaningfully testing
the one that is not.

Reuses the same geometry generator, cart=True (Cartesian Gaussian) export
convention, and QMCkl-required TREXIO patching as test_benchmarks.py, for
direct consistency with that suite's other files.
"""
import os
import trexio
from pyscf import gto, scf
from pyscf.tools import trexio as pyscf_trexio

# Import the shared geometry generator from test_benchmarks.py so benzene's
# coordinates are generated identically here and in the main scaling suite
# (same atom ordering, same deduplication tolerance) - avoids two
# independently-maintained copies of the same geometry logic drifting apart.
from test_benchmarks import generate_acene_geometry, write_xyz


def run_scf_and_export_trexio(atoms: list, h5_filepath: str, basis_name: str):
    """
    Executes RHF in PySCF and exports via pyscf.tools.trexio, then patches
    the TREXIO file with the fields QMCkl requires but PySCF's exporter does
    not always populate. Identical in structure to test_benchmarks.py's
    version of this function, kept separate here so this script has no
    runtime dependency on anything beyond the two imported helpers above.
    """
    if os.path.exists(h5_filepath):
        os.remove(h5_filepath)

    atom_pyscf = "; ".join([f"{s} {c[0]} {c[1]} {c[2]}" for s, c in atoms])

    mol = gto.M(
        atom=atom_pyscf,
        basis=basis_name,
        cart=True,       # Cartesian Gaussians - required for QMCkl's AO kernel,
                          # which enumerates angular momentum components as
                          # x^a y^b z^c (a+b+c=l), i.e. the Cartesian convention,
                          # not spherical harmonics.
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

    pyscf_trexio.to_trexio(mf, h5_filepath)

    with trexio.File(h5_filepath, 'u', trexio.TREXIO_HDF5) as f:
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
    h5_dir = "benchmark_h5_lmax_boundary"
    geom_dir = "benchmark_geometries"
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(geom_dir, exist_ok=True)

    print("=" * 68)
    print("LMAX BOUNDARY CHECK: benzene @ cc-pV5Z (introduces L=5, h-functions)")
    print("=" * 68)

    atoms = generate_acene_geometry(1)  # n_rings=1 -> benzene
    xyz_file = os.path.join(geom_dir, "benzene.xyz")
    write_xyz(xyz_file, atoms, comment="benzene (n=1) D2h")

    h5_out = os.path.join(h5_dir, "benzene_cc-pv5z.h5")
    run_scf_and_export_trexio(atoms, h5_out, basis_name="cc-pv5z")

    print("\nLMAX boundary test file ready.")


if __name__ == "__main__":
    main()
