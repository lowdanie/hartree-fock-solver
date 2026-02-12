import time
import jax
import numpy as np

import pubchempy as pcp
import slaterform as sf


def load_molecule(name: str) -> sf.Molecule:
    compounds = pcp.get_compounds(name, "name", record_type="3d")
    atoms = sf.adapters.pubchem.load_geometry(compounds[0])
    return sf.Molecule.from_geometry(atoms, basis_name="sto-3g")


def benchmark_jax(name, f, *args):
    """
    Benchmarks a jitted JAX function, separating compile time from run time.
    """
    print(f"\n--- {name} ---")

    # Compilation
    print("  Lowering...")
    start = time.time()
    lowered = f.lower(*args)
    print(f"  Done ({time.time() - start:.4f} s)")

    print("  Compiling...")
    start = time.time()
    compiled = lowered.compile()
    print(f"  Done ({time.time() - start:.4f} s)")

    hlo_text = lowered.as_text()
    line_count = len(hlo_text.splitlines())
    mem_analysis = compiled.memory_analysis()
    print(f"  HLO Line Count: {line_count}")
    print(
        f"  Temp Heap Size: {mem_analysis.temp_size_in_bytes / 1024**2:.2f} MB"
    )
