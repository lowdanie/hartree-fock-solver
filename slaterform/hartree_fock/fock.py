from typing import Callable, NamedTuple, Optional
import functools
import dataclasses

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from slaterform.basis.basis_block import BasisBlock
from slaterform.basis.operators import (
    two_electron_matrix as two_electron_matrix_op,
)
from slaterform.jax_utils.batching import BatchedTreeTuples
from slaterform.jax_utils.gather import extract_tiles_2d
from slaterform.jax_utils.scatter import add_tiles
from slaterform.integrals.coulomb import two_electron
from slaterform.symmetry import quartet as quartet_lib
from slaterform.structure.batched_basis import BatchedBasis

_BlockOperator = Callable[
    [BasisBlock, BasisBlock, BasisBlock, BasisBlock],
    jax.Array,
]


@register_pytree_node_class
@dataclasses.dataclass
class _GroupParams:
    # The stacked basis blocks for the group of batches.
    # Length: 4
    stacks: tuple[BasisBlock, ...]

    # The starting indices of each basis block in the stack with respect to
    # the full basis set.
    # Length: 4
    stack_starts: tuple[jax.Array, ...]

    # A block operator that can operate on basis blocks with a batch dimension.
    batch_operator: _BlockOperator

    # A density matrix.
    # Shape: (n_basis, n_basis)
    P: Optional[jax.Array] = None

    def tree_flatten(self):
        children = (
            self.stacks,
            self.stack_starts,
            self.P,
        )
        aux_data = (self.batch_operator,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> "_GroupParams":
        batch_operator = aux_data[0]
        stacks, stack_starts, P = children
        return cls(
            stacks=stacks,
            stack_starts=stack_starts,
            batch_operator=batch_operator,
            P=P,
        )


class _BatchData(NamedTuple):
    tuple_indices: jax.Array  # Shape (batch_size, 4)
    mask: jax.Array  # Shape (batch_size,)


_BatchStep = Callable[
    [_GroupParams, jax.Array, _BatchData], tuple[jax.Array, None]
]


def _compute_batch_integrals(
    params: _GroupParams, batch_data: _BatchData
) -> tuple[jax.Array, quartet_lib.BatchedQuartet, jax.Array]:
    """Computes integrals for a batch.

    Returns:
        integrals: A jax array of shape (batch_size, Ni, Nj, Nk, Nl)
        block_starts: A BatchedQuartet of shape (batch_size, )
        scaled_mask: A jax array of shape (batch_size, 1, 1, 1, 1)
    """
    idx = batch_data.tuple_indices

    # Gather blocks
    blocks = tuple(
        jax.tree.map(lambda x: x[idx[:, k]], params.stacks[k]) for k in range(4)
    )

    # Get start indices
    block_starts = quartet_lib.BatchedQuartet(
        params.stack_starts[k][idx[:, k]] for k in range(4)
    )

    # Compute integrals (Batch, Ni, Nj, Nk, Nl)
    integrals = params.batch_operator(*blocks)

    # Compute symmetry mask
    stabilizer_norm = quartet_lib.compute_stabilizer_norm(block_starts)
    scaled_mask = batch_data.mask / stabilizer_norm.astype(integrals.dtype)
    scaled_mask = scaled_mask[:, None, None, None, None]

    return integrals, block_starts, scaled_mask


def _matrix_step(
    params: _GroupParams, G: jax.Array, batch_data: _BatchData
) -> tuple[jax.Array, None]:
    integrals, block_starts, scaled_mask = _compute_batch_integrals(
        params, batch_data
    )

    for sigma in quartet_lib.get_symmetries():
        # Permute integrals and apply the scaled mask.
        # Transpose dims (0, 1+s0, 1+s1...)
        batch_sigma = (0,) + tuple(s + 1 for s in sigma)
        sigma_integrals = jnp.transpose(integrals, batch_sigma) * scaled_mask

        # Permute the indices
        starts_i, starts_j, starts_k, starts_l = quartet_lib.apply_permutation(
            sigma, block_starts
        )

        # Get the shapes of the permuted basis blocks
        _, _, n_j, n_k, n_l = sigma_integrals.shape

        # Coulomb update: G_ij += (ij|kl) * P_lk
        P_lk = extract_tiles_2d(
            matrix=params.P,
            row_starts=starts_l,
            col_starts=starts_k,
            n_rows=n_l,
            n_cols=n_k,
        )
        J_ij = jnp.einsum("bijkl,blk->bij", sigma_integrals, P_lk)
        G = add_tiles(
            target=G,
            tiles=J_ij,
            starts=(starts_i, starts_j),
        )

        # Exchange update: G_il -= 0.5 * (ij|kl) * P_jk
        P_jk = extract_tiles_2d(
            matrix=params.P,
            row_starts=starts_j,
            col_starts=starts_k,
            n_rows=n_j,
            n_cols=n_k,
        )
        K_il = jnp.einsum("bijkl,bjk->bil", sigma_integrals, P_jk)
        G = add_tiles(
            target=G,
            tiles=-0.5 * K_il,
            starts=(starts_i, starts_l),
        )

    return G, None


def _integrals_step(
    params: _GroupParams, V: jax.Array, batch_data: _BatchData
) -> tuple[jax.Array, None]:
    """Computes contributions to the full two-electron integral tensor."""
    integrals, block_starts, scaled_mask = _compute_batch_integrals(
        params, batch_data
    )

    for sigma in quartet_lib.get_symmetries():
        # Permute integrals
        batch_sigma = (0,) + tuple(s + 1 for s in sigma)
        sigma_integrals = jnp.transpose(integrals, batch_sigma) * scaled_mask

        # Permute starts
        sigma_starts = quartet_lib.apply_permutation(sigma, block_starts)

        # Update the integrals tensor.
        V = add_tiles(V, sigma_integrals, sigma_starts)

    return V, None


def _process_batched_tuples(
    accumulator: jax.Array,
    batched_tuples: BatchedTreeTuples,
    global_block_starts: jax.Array,
    batch_operator: _BlockOperator,
    step_fn: _BatchStep,
    P: Optional[jax.Array] = None,
) -> jax.Array:
    """Compute contributions to the two-electron matrix from batched tuples."""
    stack_starts = tuple(
        global_block_starts[idx] for idx in batched_tuples.global_tree_indices
    )
    params = _GroupParams(
        batched_tuples.stacks, stack_starts, batch_operator, P
    )
    batch_data = _BatchData(
        batched_tuples.tuple_indices, batched_tuples.padding_mask
    )
    scan_fn = functools.partial(step_fn, params)
    new_accumulator, _ = jax.lax.scan(scan_fn, accumulator, batch_data)

    return new_accumulator


def two_electron_matrix(
    basis: BatchedBasis,
    P: jax.Array,
) -> jax.Array:
    """Compute the two-electron contribution to the Fock matrix.

    Args:
        basis: The batched molecular basis.
        P: The closed shell density matrix. shape: (n_basis, n_basis)
    Returns:
        The two-electron Fock matrix. Shape: (n_basis, n_basis).
    """
    print(f"two electron matrix. num batches: {len(basis.batches_2e)}")
    n_basis = basis.n_basis
    G = jnp.zeros((n_basis, n_basis), dtype=jnp.float64)
    batch_operator = jax.vmap(
        functools.partial(two_electron_matrix_op, operator=two_electron)
    )

    for i, batched_tuples in enumerate(basis.batches_2e):
        G = _process_batched_tuples(
            G,
            batched_tuples,
            jnp.asarray(basis.block_starts),
            batch_operator,
            jax.checkpoint(_matrix_step),
            P,
        )

    return G


def two_electron_integrals(
    basis: BatchedBasis,
) -> jax.Array:
    """Compute all two-electron integrals.

    Args:
        basis: The batched molecular basis.
    Returns:
        The two-electron integral tensor.
        Shape: (n_basis, n_basis, n_basis, n_basis).
    """
    n_basis = basis.n_basis
    V = jnp.zeros((n_basis,) * 4, dtype=jnp.float64)
    batch_operator = jax.vmap(
        functools.partial(two_electron_matrix_op, operator=two_electron)
    )

    for batched_tuples in basis.batches_2e:
        V = _process_batched_tuples(
            V,
            batched_tuples,
            jnp.asarray(basis.block_starts),
            batch_operator,
            jax.checkpoint(_integrals_step),
        )

    return V


def two_electron_matrix_from_integrals(V: jax.Array, P: jax.Array) -> jax.Array:
    """Compute the two-electron contribution to the Fock matrix from the
    two-electron integral tensor.

    Args:
        V: The two-electron integral tensor.
            Shape: (n_basis, n_basis, n_basis, n_basis).
        P: The closed shell density matrix. shape: (n_basis, n_basis)
    Returns:
        The two-electron Fock matrix. Shape: (n_basis, n_basis).
    """
    # Coulomb contribution: G_ij += sum_{kl} (ij|kl) * P_lk
    J = jnp.einsum("ijkl,lk->ij", V, P)

    # Exchange contribution: G_ij -= 0.5 * (il|kj) * P_lk
    K = jnp.einsum("ilkj,lk->ij", V, P)

    G = J - 0.5 * K
    return G


def electronic_energy(
    H_core: jax.Array, F: jax.Array, P: jax.Array
) -> jax.Array:
    """Compute the electronic expectation energy from the Fock and density
    matrices.

    Formula:
        E = 0.5 * sum_{ij}P_ij(H_core_ji + F_ji)

    Args:
        H_core: The core Hamiltonian matrix. shape: (n_basis, n_basis)
        F: The core Fock matrix. shape: (n_basis, n_basis)
        P: The closed shell density matrix. shape: (n_basis, n_basis)
    Returns:
        The electronic energy. Shape: ().
    """
    # Note that P is symmetric so we can use P_ij = P_ji
    return 0.5 * jnp.sum(P * (H_core + F))
