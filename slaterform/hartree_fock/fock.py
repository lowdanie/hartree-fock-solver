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
    group_idx: jax.Array  # Shape (,)


_BatchStep = Callable[
    [_GroupParams, jax.Array, _BatchData], tuple[jax.Array, None]
]

_CapturedBatchStep = Callable[[jax.Array, _BatchData], tuple[jax.Array, None]]


def _unpack_batch_group(
    group_index: int,
    batch_group: BatchedTreeTuples,
    global_block_starts: jax.Array,
    batch_operator: _BlockOperator,
    P: Optional[jax.Array] = None,
) -> tuple[_BatchData, _GroupParams]:
    """Unpack a BatchedTreeTuples into _BatchData and _GroupParams.

    The output _BatchData has shape:
        tuple_indices: (num_batches, batch_size, 4)
        mask: (num_batches, batch_size,)
        group_idx: (num_batches,)
    """
    n_batches = batch_group.tuple_indices.shape[0]
    group_idx = jnp.full((n_batches,), group_index, dtype=jnp.int32)
    stack_starts = tuple(
        global_block_starts[idx] for idx in batch_group.global_tree_indices
    )
    params = _GroupParams(batch_group.stacks, stack_starts, batch_operator, P)
    batch_data = _BatchData(
        batch_group.tuple_indices,
        batch_group.padding_mask,
        group_idx,
    )

    return batch_data, params


def _prepare_for_scan(
    basis: BatchedBasis,
    batch_operator: _BlockOperator,
    batch_step: _BatchStep,
    P: Optional[jax.Array] = None,
) -> tuple[_BatchData, list[_CapturedBatchStep]]:
    """Prepare batched tuples for a scan over batches.

    basis.batches_2e contains a sequence of BatchedTreeTuples, each with
    a batch size of batch_size.

    The batches are combined into a single stack of _BatchData with shapes:
        tuple_indices: (total_batches, batch_size, 4)
        mask: (total_batches, batch_size,)
        group_idx: (total_batches,)

    For each group, a scan function is created that captures the corresponding
    _GroupParams and can be used to process a batch from the group.
    """
    step_fns: list[_CapturedBatchStep] = []
    batch_data_list: list[_BatchData] = []

    for group_idx, batched_tuples in enumerate(basis.batches_2e):
        batch_data, params = _unpack_batch_group(
            group_idx,
            batched_tuples,
            jnp.asarray(basis.block_starts),
            batch_operator,
            P,
        )

        # Capture params in the step function
        captured_step_fn = functools.partial(batch_step, params)
        step_fns.append(captured_step_fn)
        batch_data_list.append(batch_data)

    batch_data_all = jax.tree.map(
        lambda *args: jnp.concatenate(args, axis=0), *batch_data_list
    )

    return batch_data_all, step_fns


def _scan_with_switch(
    init_carry: jax.Array,
    batch_data: _BatchData,
    scan_fns: list[_CapturedBatchStep],
) -> jax.Array:
    def scan_body(carry: jax.Array, bd: _BatchData) -> tuple[jax.Array, None]:
        return jax.lax.switch(bd.group_idx, scan_fns, carry, bd)

    final_carry, _ = jax.lax.scan(scan_body, init_carry, batch_data)
    return final_carry


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
    n_basis = basis.n_basis
    G = jnp.zeros((n_basis, n_basis), dtype=jnp.float64)
    batch_operator = jax.vmap(
        functools.partial(two_electron_matrix_op, operator=two_electron)
    )

    batch_data, scan_fns = _prepare_for_scan(
        basis, batch_operator, jax.checkpoint(_matrix_step), P
    )
    return _scan_with_switch(G, batch_data, scan_fns)


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

    batch_data, scan_fns = _prepare_for_scan(
        basis, batch_operator, jax.checkpoint(_integrals_step)
    )
    return _scan_with_switch(V, batch_data, scan_fns)


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
