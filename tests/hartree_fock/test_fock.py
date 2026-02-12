import itertools
import functools
from collections.abc import Sequence

import pytest

from jax import jit
from jax import numpy as jnp
import numpy as np

import slaterform as sf
from slaterform.basis import operators

_TEST_MOLECULE = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0]),
            shells=[
                sf.ContractedGTO(
                    primitive_type=sf.PrimitiveType.CARTESIAN,
                    angular_momentum=(0,),
                    exponents=np.array([0.1], dtype=np.float64),
                    coefficients=np.array([[1.0]], dtype=np.float64),
                )
            ],
        ),
        sf.Atom(
            symbol="O",
            number=8,
            position=np.array([0.0, 0.0, 1.0]),
            shells=[
                sf.ContractedGTO(
                    primitive_type=sf.PrimitiveType.CARTESIAN,
                    angular_momentum=(0, 1),
                    exponents=np.array([0.2], dtype=np.float64),
                    coefficients=np.array([[0.3], [0.4]], dtype=np.float64),
                )
            ],
        ),
    ]
)

_TEST_P = np.array(
    [
        [1, 2, 3, 4, 5],
        [2, 6, 7, 8, 9],
        [3, 7, 1, 2, 3],
        [4, 8, 2, 4, 5],
        [5, 9, 3, 5, 6],
    ],
    dtype=np.float64,
)

# Computed via _brute_force_two_electron_matrix(_TEST_MOLECULE, _TEST_P)
# _EXPECTED_G = np.array(
#     [
#         [0.8258334, 0.22539037, -0.04605533, -0.05555967, -0.16525297],
#         [0.22539037, 0.08320239, -0.01676027, -0.02073222, -0.02083503],
#         [-0.04605533, -0.01676027, 0.18367568, -0.00732045, -0.00448176],
#         [-0.05555967, -0.02073222, -0.00732045, 0.172695, -0.0096359],
#         [-0.16525297, -0.02083503, -0.00448176, -0.0096359, 0.18568163],
#     ]
# )


def _build_slices(block_sizes: Sequence[int]) -> list[slice]:
    slices = []
    current = 0
    for size in block_sizes:
        slices.append(slice(current, current + size))
        current += size
    return slices


def _two_electron_integrals(basis: sf.BatchedBasis) -> np.ndarray:
    n_basis = basis.n_basis
    blocks = basis.basis_blocks
    slices = _build_slices([block.n_basis for block in blocks])

    V = np.zeros((n_basis, n_basis, n_basis, n_basis), dtype=np.float64)

    kernel = jit(
        functools.partial(
            operators.two_electron_matrix,
            operator=sf.integrals.two_electron,
        )
    )
    for i, j, k, l in itertools.product(range(len(blocks)), repeat=4):
        V_block = kernel(blocks[i], blocks[j], blocks[k], blocks[l])
        V[slices[i], slices[j], slices[k], slices[l]] = V_block

    return V


def _two_electron_matrix_from_integrals(
    V: np.ndarray, P: np.ndarray
) -> np.ndarray:
    n_basis = V.shape[0]

    # Compute G using the definition.
    # G_{ij} = sum_{kl} P_{kl} ( (ij|lk) - 0.5 (ik|lj) )
    G = np.zeros((n_basis, n_basis), dtype=np.float64)
    for i, j, k, l in itertools.product(range(n_basis), repeat=4):
        G[i, j] += P[k, l] * (V[i, j, l, k] - 0.5 * V[i, k, l, j])

    return G


_TEST_V = _two_electron_integrals(sf.BatchedBasis.from_molecule(_TEST_MOLECULE))
_TEST_G = _two_electron_matrix_from_integrals(_TEST_V, _TEST_P)


@pytest.mark.parametrize(
    "mol,P,batch_size,expected",
    [
        (_TEST_MOLECULE, _TEST_P, 1, _TEST_G),
        (_TEST_MOLECULE, _TEST_P, 2, _TEST_G),
        (_TEST_MOLECULE, _TEST_P, 2048, _TEST_G),
    ],
)
def test_two_electron_matrix(mol, P, batch_size, expected):
    basis = sf.BatchedBasis.from_molecule(mol, batch_size_2e=batch_size)
    P_jax = jnp.asarray(P)

    G_actual = jit(sf.hartree_fock.two_electron_matrix)(basis, P_jax)

    np.testing.assert_allclose(G_actual, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "mol,batch_size,expected",
    [
        (_TEST_MOLECULE, 1, _TEST_V),
        (_TEST_MOLECULE, 2, _TEST_V),
        (_TEST_MOLECULE, 2048, _TEST_V),
    ],
)
def test_two_electron_integrals(mol, batch_size, expected):
    basis = sf.BatchedBasis.from_molecule(mol, batch_size_2e=batch_size)

    V_actual = jit(
        sf.hartree_fock.two_electron_integrals, static_argnames=["dtype"]
    )(basis)

    np.testing.assert_allclose(V_actual, expected, rtol=1e-7, atol=1e-7)


def test_two_electron_integrals_f32():
    basis = sf.BatchedBasis.from_molecule(_TEST_MOLECULE, batch_size_2e=2)

    V_actual = jit(
        sf.hartree_fock.two_electron_integrals, static_argnames=["dtype"]
    )(basis, dtype=jnp.float32)

    assert V_actual.dtype == jnp.float32
    np.testing.assert_allclose(V_actual, _TEST_V, rtol=1e-7, atol=1e-7)


def test_two_electron_matrix_from_integrals(
    V=_TEST_V, P=_TEST_P, expected=_TEST_G
):
    V_jax = jnp.asarray(V)
    P_jax = jnp.asarray(P)
    G_actual = jit(sf.hartree_fock.two_electron_matrix_from_integrals)(
        V_jax, P_jax
    )
    np.testing.assert_allclose(G_actual, expected, rtol=1e-7, atol=1e-7)


def test_electronic_energy():
    H_core = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )

    F = np.array(
        [
            [1, 6, 7],
            [2, 5, 8],
            [3, 4, 9],
        ],
        dtype=np.float64,
    )

    P = np.array(
        [
            [9, 8, 7],
            [8, 6, 5],
            [7, 5, 4],
        ],
        dtype=np.float64,
    )

    # Compute the expectation energy directly.
    # E = 0.5 * sum_{ij}P_ij(H_core_ji + F_ji)
    E_expected = 0.0
    for i, j in itertools.product(range(3), repeat=2):
        E_expected += P[i, j] * (H_core[j, i] + F[j, i])
    E_expected *= 0.5

    E_result = jit(sf.hartree_fock.electronic_energy)(H_core, F, P)

    np.testing.assert_almost_equal(E_result, E_expected, decimal=10)
