import jax

jax.config.update("jax_enable_x64", True)

import pubchempy as pcp


import slaterform as sf
from benchmarks import utils

mol = utils.load_molecule("water")
basis = sf.BatchedBasis.from_molecule(mol, batch_size_2e=32)

print(f"Total basis functions: {basis.n_basis}")
print("Batches for 2-electron integrals:")
for batches in basis.batches_2e:
    print(batches.tuple_indices.shape)

utils.benchmark_jax(
    "Two-Electron Integrals (Water)",
    sf.hartree_fock.two_electron_integrals,
    basis,
)
