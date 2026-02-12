import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import slaterform as sf
import slaterform.hartree_fock.scf as scf
from benchmarks import utils

MOLECULE_NAME = "Water"
BATCH_SIZE_1E = 16
BATCH_SIZE_2E = 32

mol = utils.load_molecule(MOLECULE_NAME)
basis = sf.BatchedBasis.from_molecule(
    mol, batch_size_1e=BATCH_SIZE_1E, batch_size_2e=BATCH_SIZE_2E
)
options = scf.Options(
    integral_strategy=scf.CachedStrategy(dtype=jnp.float64),
)

print(f"Total basis functions: {basis.n_basis}")
print("Batches for 2-electron integrals:")
for batches in basis.batches_2e:
    print(batches.tuple_indices.shape)

utils.benchmark_jax(
    f"Two-Electron Integrals ({MOLECULE_NAME})",
    sf.hartree_fock.two_electron_integrals,
    basis,
)
