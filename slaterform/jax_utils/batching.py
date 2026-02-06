import dataclasses
from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar, Hashable

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

# A pytree type.
Tree: TypeAlias = Any


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class BatchedTreeTuples:
    """A batch of pytree tuples for jax processing.

    The trees in each stack all have the same static signature.
    """

    # stacks[i] is a single pytree where the leaves are stacked arrays
    # with shape (n_items[i], ...).
    # Length: tuple_size
    stacks: tuple[Tree, ...]

    # Indices of the pytrees in the original input sequence.
    # global_tree_indices[i] is an array of shape (n_items[i],)
    # Length: tuple_size
    global_tree_indices: tuple[jax.Array, ...]

    # Batches of tree tuple indices. The tuple indices index into the
    # tree stacks.
    # Shape (n_batches, batch_size, tuple_size)
    tuple_indices: jax.Array

    # Mask indicating which entries in the batches are valid.
    # Shape (n_batches, batch_size)
    padding_mask: jax.Array

    def tree_flatten(self):
        children = (
            self.stacks,
            self.global_tree_indices,
            self.tuple_indices,
            self.padding_mask,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[
            tuple[Tree, ...],
            tuple[jax.Array, ...],
            jax.Array,
            jax.Array,
        ],
    ) -> "BatchedTreeTuples":
        return cls(
            stacks=children[0],
            global_tree_indices=children[1],
            tuple_indices=children[2],
            padding_mask=children[3],
        )


@dataclasses.dataclass(frozen=True, order=True)
class TreeSignature:
    aux_data: Hashable
    leaf_shapes: tuple[tuple[int, ...], ...]


_GroupKey = tuple[TreeSignature, ...]


@dataclasses.dataclass(frozen=True)
class _Group:
    """A group of pytrees with the same signature."""

    # trees[i] is dict of trees keyed with their global index.
    # Length: tuple_size
    trees: tuple[dict[int, Tree], ...]

    # Each tuple has size tuple_size. The indices are into the global
    # list of trees.
    tuple_indices: list[tuple[int, ...]]


def compute_tree_signature(
    tree: Any,
) -> TreeSignature:
    """Computes a static signature for a BasisBlock."""
    leaves, aux_data = jax.tree.flatten(tree)
    leaf_shapes = tuple(jnp.shape(leaf) for leaf in leaves)
    return TreeSignature(aux_data=aux_data, leaf_shapes=leaf_shapes)


def _compute_group_key(
    trees: Sequence[Tree],
    tuple_index: tuple[int, ...],
) -> _GroupKey:
    return tuple(compute_tree_signature(trees[i]) for i in tuple_index)


def _init_group(
    tuple_size: int,
) -> _Group:
    return _Group(
        trees=tuple(dict() for _ in range(tuple_size)),
        tuple_indices=[],
    )


def _update_group(
    group: _Group,
    trees: Sequence[Tree],
    tuple_index: tuple[int, ...],
) -> None:
    group.tuple_indices.append(tuple_index)

    # Update the trees if they are not already present.
    for i, tree_idx in enumerate(tuple_index):
        if tree_idx not in group.trees[i]:
            group.trees[i][tree_idx] = trees[tree_idx]


def _group_tree_tuples(
    trees: Sequence[Tree],
    tuple_length: int,
    tuple_indices: Sequence[tuple[int, ...]],
) -> dict[_GroupKey, _Group]:
    """Group the tree tuples by their static signatures."""
    groups: dict[_GroupKey, _Group] = {}

    for idx_tuple in tuple_indices:
        group_key = _compute_group_key(trees, idx_tuple)
        if group_key not in groups:
            groups[group_key] = _init_group(tuple_length)
        _update_group(groups[group_key], trees, idx_tuple)

    return groups


def _generate_tree_stack(
    trees: dict[int, Tree],
) -> tuple[Tree, np.ndarray]:
    """Stacks the pytrees into a single pytree and records the global indices."""
    # We're relying on the fact that dicts preserve insertion order.
    stack = list(trees.values())
    global_indices = np.array(list(trees.keys()))

    # Stack the arrays in the basis blocks
    stack = jax.tree.map(lambda *xs: jnp.stack(xs), *stack)
    return stack, global_indices


def _batch_tuple_indices(
    tuple_indices: np.ndarray, batch_size: int
) -> tuple[jax.Array, jax.Array]:
    """Batch tuple indices with padding."""
    n_tuples = tuple_indices.shape[0]
    tuple_size = tuple_indices.shape[1]

    remainder = n_tuples % batch_size
    pad_size = (batch_size - remainder) if remainder != 0 else 0

    # Pad the tuple indices to make n_tuples divisible by batch_size
    # shape: (a multiple of batch_size, tuple_size)
    padded_tuple_indices = np.pad(
        tuple_indices,
        ((0, pad_size), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # Create a mask indicating the entries that are not padding.
    # shape: (a multiple of batch_size,)
    mask = np.concatenate([np.ones(n_tuples), np.zeros(pad_size)])

    batched_tuple_indices = padded_tuple_indices.reshape(
        -1, batch_size, tuple_size
    )
    batched_mask = mask.reshape(-1, batch_size)

    return jnp.array(batched_tuple_indices), jnp.array(batched_mask)


def _map_to_local_tuple_indices(
    global_tuple_indices: np.ndarray,
    global_tree_indices: Sequence[np.ndarray],
) -> np.ndarray:
    """Maps global tuple indices to local tuple indices.

    Args:
        global_tuple_indices: shape (n_tuples, tuple_size)
        global_tree_indices: Length tuple_size, each of shape (n_trees_i,)
    Returns:
        local_tuple_indices: shape (n_tuples, tuple_size)
    """
    # shape: (n_tuples, tuple_size)
    local_tuple_indices = np.zeros_like(global_tuple_indices)
    for col, global_indices in enumerate(global_tree_indices):
        max_global_idx = np.max(global_indices)
        lookup_table = np.zeros(max_global_idx + 1, dtype=np.int32)
        lookup_table[global_indices] = np.arange(
            len(global_indices), dtype=np.int32
        )
        local_tuple_indices[:, col] = lookup_table[global_tuple_indices[:, col]]

    return local_tuple_indices


def _batch_group(group: _Group, batch_size: int) -> BatchedTreeTuples:
    # Generate tuple_size stacks and global indices
    stacks = []
    global_block_indices = []

    for tree_dict in group.trees:
        stack, global_idx = _generate_tree_stack(tree_dict)
        stacks.append(stack)
        global_block_indices.append(global_idx)

    # Convert to local tuple indices and batch them.
    global_tuple_indices = np.array(group.tuple_indices, dtype=np.int32)
    local_tuple_indices = _map_to_local_tuple_indices(
        global_tuple_indices, global_block_indices
    )
    batched_tuple_indices, padding_mask = _batch_tuple_indices(
        local_tuple_indices, batch_size
    )

    return BatchedTreeTuples(
        stacks=tuple(stacks),
        global_tree_indices=tuple(jnp.array(g) for g in global_block_indices),
        tuple_indices=jnp.array(batched_tuple_indices),
        padding_mask=jnp.array(padding_mask),
    )


def batch_tree_tuples(
    trees: Sequence[Tree],
    tuple_length: int,
    tuple_indices: Sequence[tuple[int, ...]],
    batch_size: int,
) -> Sequence[BatchedTreeTuples]:
    """Generates batches of tree tuples from the given trees and tuple indices.

    Note:
        The batched block tuples are grouped by their static signatures for efficient
        jax processing. If possible, sort the trees or tuple indices to minimize
        the number of unique tree tuple signatures before calling this function.
        For example, if the indices in each tuple are monotonic, sort the trees
        by compute_tree_signature.

    Args:
        trees: A sequence of pytrees of length n_trees.
        tuple_length: The length of each tree tuple.
        tuple_indices: A sequence of tuples of indices. Each tuple has length
            tuple_length. The indices in each tuple are between 0 and n_trees-1.
        batch_size: The batch size for batching the tuples.

    Returns:
        A sequence of BatchedTrees, one for each unique block tuple signature.
        The set of tuples in the union of the BatchedTrees are equal to the set
        of input tree tuples.
    """
    groups = _group_tree_tuples(trees, tuple_length, tuple_indices)

    return [_batch_group(g, batch_size) for g in groups.values()]
