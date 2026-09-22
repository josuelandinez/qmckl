#!/usr/bin/env python3
"""
Patches the missing TREXIO "qmc" group (qmc_num, qmc_point) into the
existing benchmark_h5_scaling_size / benchmark_h5_scaling_basis /
benchmark_h5_lmax_boundary .h5 files, IN PLACE, without regenerating them.

Why this group is missing: confirmed directly by reading test_benchmarks.py,
test_acenes.py, and test_trexio_benzene_5z.py - none of them call
trexio.write_qmc_num / write_qmc_point anywhere. This isn't something that
broke; that group was simply never written by these generators, since their
purpose was AO/MO scaling data, not QMC walker data. bench_aos.c reads this
group directly via trexio_read_qmc_num/trexio_read_qmc_point, so its absence
is what produced the "-nan" in that file's own Non-zeros diagnostic print.

SAFETY, in order of priority (explicitly requested - these files must not be
put at risk):
  1. Defaults to --dry-run: prints exactly what it would do, writes nothing,
     until you explicitly pass --apply.
  2. Every file gets a one-time backup (<name>.h5.bak) before the FIRST time
     it is ever modified. If a .bak already exists, it is never overwritten
     or touched again - it always reflects the untouched original, even if
     this script is run multiple times across multiple sessions.
  3. Idempotent: any file that already has both qmc_num and qmc_point is
     skipped entirely (reported, not modified) - safe to re-run this script
     as many times as needed without duplicating or overwriting data.
  4. Opens files in TREXIO's own 'u' (update) mode - the same mode the
     generation scripts themselves use for their own post-SCF patch step -
     never recreates or truncates a file.
  5. Reads electron_num and nucleus_coord directly from each file (never
     hardcoded), so walker generation is specific to that file's own
     molecule - never assumes counts from a different file.

What gets written, and why these specific choices:
  - qmc_num = 100 walkers: matches the established convention already used
    by Alz_small.h5 / Alz_large.h5 (qmc_num=100), for consistency with the
    reference files this project has used throughout.
  - electron_num is read from the file itself (varies per molecule - e.g.
    benzene has far fewer electrons than tetracene).
  - Positions: for each walker, each electron is placed as a Gaussian
    displacement (sigma=2.5 Bohr, chosen after directly inspecting
    Alz_large.h5's own qmc_point values - see DISPLACEMENT_SIGMA below)
    around a randomly-selected nucleus of THAT molecule, using a fixed seed
    (fixed per file, derived from its own path) so re-running this script
    reproduces identical positions rather than regenerating different data
    on every run. This is not a converged VMC walker distribution - it does
    not need to be for benchmarking AO/MO kernel throughput, which only
    needs electrons at plausible, non-degenerate distances from the nuclei
    so the kernel's near/far cutoff logic is genuinely exercised, the same
    property overlap_aos.c's own bounding-box approach relies on.
"""
import argparse
import hashlib
import os
import shutil
import sys

import numpy as np
import trexio

# All three benchmark_h5_* directories under the project root, as given.
BASE_DIR = "/p/project1/numeriqs/landinezborda1/B2/QMCkl_kokkos"
TARGET_DIRS = [
    "benchmark_h5_scaling_size",
    "benchmark_h5_scaling_basis",
    "benchmark_h5_lmax_boundary",
]

WALK_NUM = 100          # matches Alz_small.h5 / Alz_large.h5's own qmc_num
DISPLACEMENT_SIGMA = 2.5  # Bohr - widened after directly inspecting Alz_large's
                          # own qmc_point values: real VMC walker positions span
                          # a range comparable to the molecule's full spatial
                          # extent (e.g. observed values like -14.9, 8.6, 5.8 for
                          # a molecule whose nuclei span roughly -10 to +14 Bohr),
                          # not tight ~1 Bohr clustering around a single nucleus.
                          # 2.5 Bohr keeps generation just as simple while being a
                          # closer match to that observed spread.


def find_h5_files():
    files = []
    for d in TARGET_DIRS:
        full_dir = os.path.join(BASE_DIR, d)
        if not os.path.isdir(full_dir):
            print(f"  [skip] directory not found: {full_dir}")
            continue
        for name in sorted(os.listdir(full_dir)):
            if name.endswith(".h5"):
                files.append(os.path.join(full_dir, name))
    return files


def seed_for_file(path):
    # Fixed, file-specific seed derived from the path itself, so re-running
    # this script always reproduces identical walker positions for a given
    # file, rather than generating different random data on every run.
    h = hashlib.sha256(path.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


def already_patched(h5_path):
    with trexio.File(h5_path, 'r', trexio.TREXIO_HDF5) as f:
        return trexio.has_qmc_num(f) and trexio.has_qmc_point(f)


def generate_walker_points(nucleus_coord, electron_num, walk_num, seed):
    """nucleus_coord: (nucl_num, 3) array, already in the file's own units
    (Bohr, matching how QMCkl/TREXIO stores it). Returns (walk_num,
    electron_num, 3)."""
    rng = np.random.default_rng(seed)
    nucl_num = nucleus_coord.shape[0]
    points = np.empty((walk_num, electron_num, 3), dtype=np.float64)
    for w in range(walk_num):
        # Each electron placed near a randomly-chosen nucleus (with
        # replacement - electron_num need not equal nucl_num).
        chosen = rng.integers(0, nucl_num, size=electron_num)
        displacement = rng.normal(loc=0.0, scale=DISPLACEMENT_SIGMA, size=(electron_num, 3))
        points[w] = nucleus_coord[chosen] + displacement
    return points


def process_file(h5_path, apply_changes):
    print(f"\n{h5_path}")

    if already_patched(h5_path):
        print("  [skip] already has qmc_num and qmc_point - not touching")
        return "skipped_already_patched"

    with trexio.File(h5_path, 'r', trexio.TREXIO_HDF5) as f:
        electron_num = trexio.read_electron_num(f)
        nucleus_coord = np.array(trexio.read_nucleus_coord(f)).reshape(-1, 3)

    print(f"  electron_num={electron_num}  nucl_num={nucleus_coord.shape[0]}")
    print(f"  would write: qmc_num={WALK_NUM}, "
          f"qmc_point shape=({WALK_NUM}, {electron_num}, 3)")

    if not apply_changes:
        print("  [dry-run] no changes written")
        return "dry_run"

    bak_path = h5_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(h5_path, bak_path)
        print(f"  [backup] created {bak_path}")
    else:
        print(f"  [backup] already exists, left untouched: {bak_path}")

    seed = seed_for_file(h5_path)
    points = generate_walker_points(nucleus_coord, electron_num, WALK_NUM, seed)

    with trexio.File(h5_path, 'u', trexio.TREXIO_HDF5) as f:
        trexio.write_qmc_num(f, WALK_NUM)
        trexio.write_qmc_point(f, points)

    print(f"  [applied] wrote qmc_num={WALK_NUM} and qmc_point (seed={seed})")
    return "patched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                         help="Actually write changes. Without this flag, "
                              "only reports what would be done.")
    args = parser.parse_args()

    if not args.apply:
        print("=" * 70)
        print(" DRY RUN - no files will be modified.")
        print(" Re-run with --apply once this output looks correct.")
        print("=" * 70)

    files = find_h5_files()
    if not files:
        print("No .h5 files found under the target directories - nothing to do.")
        sys.exit(0)

    print(f"Found {len(files)} .h5 file(s) to check.")

    results = {}
    for path in files:
        try:
            status = process_file(path, apply_changes=args.apply)
        except Exception as e:
            print(f"  [ERROR] {path}: {e}")
            status = "error"
        results[path] = status

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for path, status in results.items():
        print(f"  {status:24s} {path}")


if __name__ == "__main__":
    main()
