from collections import Counter
import pytest

from jax import jit
import jax.numpy as jnp
import numpy as np

import slaterform as sf
from tests.jax_utils import block_utils
from tests.jax_utils import pytree_utils

_S_SHELL = sf.ContractedGTO(
    primitive_type=sf.PrimitiveType.CARTESIAN,
    angular_momentum=(0,),
    exponents=jnp.array([1.0]),
    coefficients=jnp.array([1.0]),
)

_SP_SHELL = sf.ContractedGTO(
    primitive_type=sf.PrimitiveType.CARTESIAN,
    angular_momentum=(0, 1),
    exponents=jnp.array([1.0, 0.5]),
    coefficients=jnp.array([[1.0, 0.5], [0.8, 0.3]]),
)

# block sizes: [1, 1]
_TEST_MOLECULE_H2 = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=jnp.array([0.0, 0.0, 0.0]),
            shells=[_S_SHELL],
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=jnp.array([0.0, 0.0, 1.4]),
            shells=[_S_SHELL],
        ),
    ]
)

# block sizes: [4, 1]
_TEST_MOLECULE_HO = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="O",
            number=8,
            position=jnp.array([0.0, 0.0, 1.0]),
            shells=[_SP_SHELL],
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=jnp.array([0.0, 0.0, 0.0]),
            shells=[_S_SHELL],
        ),
    ]
)

# block sizes: [1, 1, 4]
_TEST_MOLECULE_H2O = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=jnp.array([1.0, 0.0, 0.0]),
            shells=[_S_SHELL],
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=jnp.array([0.0, 1.0, 0.0]),
            shells=[_S_SHELL],
        ),
        sf.Atom(
            symbol="O",
            number=8,
            position=jnp.array([0.0, 0.0, 0.0]),
            shells=[_SP_SHELL],
        ),
    ]
)


def test_dtype_empty():
    basis = sf.BatchedBasis(
        atoms=[],
        basis_blocks=[],
        block_starts=jnp.array([], dtype=np.int32),
        block_atom_indices=jnp.array([], dtype=np.int32),
        batches_1e=[],
        batches_2e=[],
    )
    assert basis.dtype == jnp.float64


def test_dtype_f64():
    basis = jit(sf.BatchedBasis.from_molecule)(_TEST_MOLECULE_H2O)
    assert basis.dtype == jnp.float64


def test_dtype_f32():
    basis = jit(sf.BatchedBasis.from_molecule)(_TEST_MOLECULE_H2O)
    basis.basis_blocks[0].center = basis.basis_blocks[0].center.astype(
        jnp.float32
    )
    assert basis.dtype == jnp.float32


def test_atoms():
    basis = jit(sf.BatchedBasis.from_molecule)(_TEST_MOLECULE_H2O)

    assert len(basis.atoms) == 3
    for actual, expected in zip(basis.atoms, _TEST_MOLECULE_H2O.atoms):
        pytree_utils.assert_pytrees_equal(actual, expected)


def test_block_atom_indices():
    basis = jit(sf.BatchedBasis.from_molecule)(_TEST_MOLECULE_H2O)
    expected = np.array([2, 0, 1], dtype=np.int32)
    np.testing.assert_array_equal(basis.block_atom_indices, expected)


@pytest.mark.parametrize(
    "molecule,expected",
    [
        (
            _TEST_MOLECULE_H2,
            [0, 1],
        ),
        (
            _TEST_MOLECULE_HO,
            [0, 4],
        ),
        (
            _TEST_MOLECULE_H2O,
            [0, 4, 5],
        ),
    ],
)
def test_block_starts(molecule, expected):
    basis = sf.BatchedBasis.from_molecule(molecule)
    np.testing.assert_array_equal(basis.block_starts, expected)


@pytest.mark.parametrize(
    "molecule,expected_tuple_batches",
    [
        (
            _TEST_MOLECULE_H2,
            [
                [(0, 0), (0, 1)],
                [(1, 1)],
            ],
        ),
        (
            _TEST_MOLECULE_HO,
            [
                [(0, 0)],
                [(0, 1)],
                [(1, 1)],
            ],
        ),
        (
            _TEST_MOLECULE_H2O,
            [
                [(0, 0)],  # SP,SP
                [(0, 1), (0, 2)],  # SP,S
                [(1, 1), (1, 2)],  # S,S
                [(2, 2)],  # S,S
            ],
        ),
    ],
)
def test_1e_tuples(molecule, expected_tuple_batches):
    basis = sf.BatchedBasis.from_molecule(molecule, batch_size_1e=2)

    expected = [tuple(b) for b in expected_tuple_batches]
    actual = []
    for batched_tuples in basis.batches_1e:
        actual.extend(
            map(tuple, block_utils.get_global_tuple_batches(batched_tuples))
        )

    assert Counter(actual) == Counter(expected)


@pytest.mark.parametrize(
    "molecule,expected",
    [
        (
            _TEST_MOLECULE_H2,
            [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (1, 0, 1, 0),
                (1, 1, 0, 0),
                (1, 1, 1, 0),
                (1, 1, 1, 1),
            ],
        ),
        (
            _TEST_MOLECULE_HO,
            [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (1, 0, 1, 0),
                (1, 1, 0, 0),
                (1, 1, 1, 0),
                (1, 1, 1, 1),
            ],
        ),
    ],
)
def test_2e_tuples(molecule, expected):
    basis = sf.BatchedBasis.from_molecule(molecule, batch_size_2e=2)

    actual = []
    for batched_tuples in basis.batches_2e:
        assert batched_tuples.tuple_indices.shape[1] <= 2  # batch size
        for batch in block_utils.get_global_tuple_batches(batched_tuples):
            actual.extend(batch)

    assert Counter(actual) == Counter(expected)


@pytest.mark.parametrize(
    "molecule",
    [
        _TEST_MOLECULE_H2,
        _TEST_MOLECULE_HO,
        _TEST_MOLECULE_H2O,
    ],
)
def test_with_positions(molecule: sf.Molecule):
    basis = sf.BatchedBasis.from_molecule(molecule)
    new_positions = jnp.array([atom.position for atom in molecule.atoms]) + 1.0
    new_basis = basis.with_positions(new_positions)

    for i, atom in enumerate(new_basis.atoms):
        np.testing.assert_array_equal(atom.position, new_positions[i])

    for i, block in enumerate(new_basis.basis_blocks):
        expected_center = new_positions[basis.block_atom_indices[i]]
        np.testing.assert_array_equal(block.center, expected_center)

    for batch in new_basis.batches_1e:
        for stack, global_indices in zip(
            batch.stacks, batch.global_tree_indices
        ):
            for center, global_idx in zip(stack.center, global_indices):
                atom_idx = basis.block_atom_indices[global_idx]
                expected_center = new_positions[atom_idx]
                np.testing.assert_array_equal(center, expected_center)

    for batch in new_basis.batches_2e:
        for stack, global_indices in zip(
            batch.stacks, batch.global_tree_indices
        ):
            for center, global_idx in zip(stack.center, global_indices):
                atom_idx = basis.block_atom_indices[global_idx]
                expected_center = new_positions[atom_idx]
                np.testing.assert_array_equal(center, expected_center)
