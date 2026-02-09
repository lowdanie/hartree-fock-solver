import dataclasses
import itertools
from collections.abc import Sequence

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from slaterform.basis.basis_block import BasisBlock
from slaterform.jax_utils.batching import BatchedTreeTuples, batch_tree_tuples
from slaterform.structure.atom import Atom
from slaterform.structure.molecule import Molecule
from slaterform.symmetry.quartet import iter_canonical_quartets


def _build_basis_blocks(
    atoms: Sequence[Atom],
) -> tuple[list[BasisBlock], list[int]]:
    blocks: list[BasisBlock] = []
    indices: list[int] = []
    for i, atom in enumerate(atoms):
        atom_blocks = [
            BasisBlock.from_gto(gto, atom.position) for gto in atom.shells
        ]
        blocks.extend(atom_blocks)
        indices.extend([i] * len(atom_blocks))

    paired = zip(blocks, indices)
    sorted_paired = sorted(
        paired, key=lambda pair: pair[0].n_cart, reverse=True
    )
    sorted_blocks, sorted_indices = zip(*sorted_paired)
    return list(sorted_blocks), list(sorted_indices)


def _update_batch_centers(
    batch: BatchedTreeTuples, global_block_centers: jax.Array
) -> BatchedTreeTuples:
    """Returns a new BatchedTreeTuples with updated centers."""
    new_stacks = []
    for stack, global_indices in zip(batch.stacks, batch.global_tree_indices):
        # Gather the new centers for this batch
        # shape: (batch_size, 3)
        new_stack_centers = global_block_centers[global_indices]
        new_stack = dataclasses.replace(stack, center=new_stack_centers)
        new_stacks.append(new_stack)

    return dataclasses.replace(batch, stacks=tuple(new_stacks))


@register_pytree_node_class
@dataclasses.dataclass
class BatchedBasis:
    """A Molecule with pre-computed batched basis structures for integration.

    Holds both 1-electron (pair) and 2-electron batches.
    """

    atoms: Sequence[Atom]
    basis_blocks: Sequence[BasisBlock]

    # The starting indices of each basis block in the full basis set.
    # Shape: (n_blocks,)
    block_starts: jax.Array

    # The indices of the atoms corresponding to each basis block.
    # Shape: (n_blocks,)
    block_atom_indices: jax.Array

    # Batches for 1-electron integrals. Tuple length = 2.
    batches_1e: Sequence[BatchedTreeTuples]

    # Batches for 2-electron integrals. Tuple length = 4.
    batches_2e: Sequence[BatchedTreeTuples]

    @property
    def n_basis(self) -> int:
        """The total number of basis functions in this molecular basis."""
        return sum(block.n_basis for block in self.basis_blocks)

    @property
    def n_electrons(self) -> int:
        """The total number of electrons in the molecule."""
        return sum(atom.number for atom in self.atoms)

    def tree_flatten(self):
        children = (
            self.atoms,
            self.basis_blocks,
            self.block_starts,
            self.block_atom_indices,
            self.batches_1e,
            self.batches_2e,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        batch_size_1e: int = 256,
        batch_size_2e: int = 64,
    ) -> "BatchedBasis":
        basis_blocks, block_atom_indices = _build_basis_blocks(molecule.atoms)

        n_blocks = len(basis_blocks)
        block_sizes = np.array([block.n_basis for block in basis_blocks])
        block_starts = jnp.array(
            np.concatenate(([0], np.cumsum(block_sizes)[:-1])), dtype=jnp.int32
        )

        # (i, j) where 0 <= i <= j < n_blocks
        pairs = list(
            itertools.combinations_with_replacement(range(n_blocks), 2)
        )
        batches_1e = batch_tree_tuples(
            trees=basis_blocks,
            tuple_length=2,
            tuple_indices=pairs,
            batch_size=batch_size_1e,
        )

        batches_2e = batch_tree_tuples(
            trees=basis_blocks,
            tuple_length=4,
            tuple_indices=list(iter_canonical_quartets(n_blocks)),
            batch_size=batch_size_2e,
        )

        return cls(
            molecule.atoms,
            basis_blocks,
            block_starts,
            jnp.asarray(block_atom_indices),
            batches_1e,
            batches_2e,
        )

    def with_positions(self, positions: jax.Array) -> "BatchedBasis":
        """Returns a new BatchedBasis with updated atomic positions."""
        atoms = [
            atom.with_position(pos) for atom, pos in zip(self.atoms, positions)
        ]

        block_centers = positions[self.block_atom_indices]
        basis_blocks = [
            block.with_center(center)
            for block, center in zip(self.basis_blocks, block_centers)
        ]

        batches_1e = [
            _update_batch_centers(batch, block_centers)
            for batch in self.batches_1e
        ]
        batches_2e = [
            _update_batch_centers(batch, block_centers)
            for batch in self.batches_2e
        ]

        return dataclasses.replace(
            self,
            atoms=atoms,
            basis_blocks=basis_blocks,
            batches_1e=batches_1e,
            batches_2e=batches_2e,
        )
