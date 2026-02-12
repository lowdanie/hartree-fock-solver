from collections.abc import Sequence
from typing import Callable, NamedTuple
import functools

import jax
from jax import numpy as jnp


from slaterform.integrals.gaussian import GaussianBasis3d
from slaterform.integrals.overlap import overlap_3d
from slaterform.integrals.kinetic import kinetic_3d
from slaterform.integrals.coulomb import one_electron as coulomb_one_electron
from slaterform.structure.batched_basis import BatchedBasis
from slaterform.basis.basis_block import BasisBlock
from slaterform.basis.operators import OneElectronOperator, one_electron_matrix
from slaterform.jax_utils.batching import BatchedTreeTuples
from slaterform.jax_utils.scatter import add_tiles

_BlockOperator = Callable[[BasisBlock, BasisBlock], jax.Array]


class _GroupParams(NamedTuple):
    # The stacked basis blocks for the group of batches.
    # Length: 2
    stacks: tuple[BasisBlock, ...]

    # The starting indices of each basis block in the stack with respect to
    # the full basis set.
    # Length: 2
    stack_starts: tuple[jax.Array, ...]

    # A block operator that can operate on basis blocks with a batch dimension.
    batch_operator: _BlockOperator


class _BatchData(NamedTuple):
    tuple_indices: jax.Array  # Shape (batch_size, 2)
    mask: jax.Array  # Shape (batch_size,)


def _batch_step(
    params: _GroupParams, matrix: jax.Array, batch_data: _BatchData
) -> tuple[jax.Array, None]:
    # Indices of each pair of blocks in the batch.
    # shape: (batch_size,)
    i_idx = batch_data.tuple_indices[:, 0]
    j_idx = batch_data.tuple_indices[:, 1]

    # Gather the basis blocks for each pair in the batch.
    # Each has shape (batch_size, ...)
    i_block = jax.tree.map(lambda x: x[i_idx], params.stacks[0])
    j_block = jax.tree.map(lambda x: x[j_idx], params.stacks[1])

    i_start = params.stack_starts[0][i_idx]
    j_start = params.stack_starts[1][j_idx]

    # Compute the one-electron integrals for the batch.
    # shape: (batch_size, n_i_basis, n_j_basis)
    integral_matrix = params.batch_operator(i_block, j_block)

    new_matrix = add_tiles(
        target=matrix,
        tiles=integral_matrix,
        starts=(i_start, j_start),
        mask=batch_data.mask,
    )

    # Also add the transpose for the symmetric entry if i!=j
    off_diagonal_mask = batch_data.mask * (i_start != j_start).astype(
        matrix.dtype
    )

    new_matrix = add_tiles(
        target=new_matrix,
        tiles=integral_matrix.transpose((0, 2, 1)),
        starts=(j_start, i_start),
        mask=off_diagonal_mask,
    )

    return new_matrix, None


def _process_batched_tuples(
    matrix: jax.Array,
    batched_tuples: BatchedTreeTuples,
    block_starts: jax.Array,
    batch_operator: _BlockOperator,
) -> jax.Array:
    """Compute contributions to the one-electron matrix from batched tuples."""
    stack_starts = tuple(
        block_starts[idx] for idx in batched_tuples.global_tree_indices
    )
    params = _GroupParams(
        stacks=batched_tuples.stacks,
        stack_starts=stack_starts,
        batch_operator=batch_operator,
    )
    batch_data = _BatchData(
        tuple_indices=batched_tuples.tuple_indices,
        mask=batched_tuples.padding_mask,
    )
    scan_fn = functools.partial(_batch_step, params)
    new_matrix, _ = jax.lax.scan(scan_fn, matrix, batch_data)

    return new_matrix


def _one_electron_matrix(
    basis: BatchedBasis,
    operator: OneElectronOperator,
) -> jax.Array:
    n_basis = basis.n_basis
    matrix = jnp.zeros((n_basis, n_basis), dtype=basis.dtype)
    batch_operator = jax.vmap(
        lambda b1, b2: one_electron_matrix(b1, b2, operator)
    )

    for batched_tuples in basis.batches_1e:
        matrix = _process_batched_tuples(
            matrix,
            batched_tuples,
            jnp.asarray(basis.block_starts),
            batch_operator,
        )

    return matrix


def overlap_matrix(
    batched_mol_basis: BatchedBasis,
) -> jax.Array:
    """Computes the overlap matrix S

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    S = _one_electron_matrix(
        batched_mol_basis,
        overlap_3d,
    )

    return S


def _nuclear_operator(
    atomic_positions: jax.Array,
    atomic_charges: jax.Array,
    g1: GaussianBasis3d,
    g2: GaussianBasis3d,
) -> jax.Array:
    tensors = jax.vmap(
        lambda R, Z: -Z * coulomb_one_electron(g1, g2, R),
        in_axes=(0, 0),
    )(atomic_positions, atomic_charges)

    return jnp.sum(tensors, axis=0)


def nuclear_attraction_matrix(
    basis: BatchedBasis,
) -> jax.Array:
    """Computes the nuclear attraction matrix V

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    positions = jnp.asarray([atom.position for atom in basis.atoms])
    charges = jnp.asarray([atom.number for atom in basis.atoms])
    nuclear_op = functools.partial(_nuclear_operator, positions, charges)

    V = _one_electron_matrix(basis, nuclear_op)

    return V


def kinetic_matrix(
    basis: BatchedBasis,
) -> jax.Array:
    """Computes the kinetic energy matrix T

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    return -0.5 * _one_electron_matrix(basis, kinetic_3d)


def core_hamiltonian_matrix(
    basis: BatchedBasis,
) -> jax.Array:
    """Computes the core Hamiltonian matrix H = T + V

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    T = kinetic_matrix(basis)
    V = nuclear_attraction_matrix(basis)

    return T + V
