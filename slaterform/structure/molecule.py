from collections.abc import Sequence
import dataclasses

import jax
from jax.tree_util import register_pytree_node_class

from slaterform.structure.atom import Atom
from slaterform.adapters.bse import load as load_basis


@register_pytree_node_class
@dataclasses.dataclass
class Molecule:
    atoms: Sequence[Atom]

    def tree_flatten(self):
        children = (self.atoms,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Sequence[Atom],]
    ) -> "Molecule":
        return cls(atoms=children[0])

    @classmethod
    def from_geometry(
        cls, atoms: Sequence[Atom], basis_name: str
    ) -> "Molecule":
        """Builds a Molecule object from a sequence of atomic positions and a basis set name."""
        atoms = [
            Atom(
                symbol=atom.symbol,
                number=atom.number,
                position=atom.position,
                shells=load_basis(basis_name=basis_name, element=atom.number),
            )
            for atom in atoms
        ]

        return cls(atoms=atoms)

    def with_positions(self, new_positions: jax.Array) -> "Molecule":
        """Returns a new Molecule with updated atomic positions."""
        new_atoms = [
            dataclasses.replace(atom, position=pos)
            for atom, pos in zip(self.atoms, new_positions)
        ]
        return dataclasses.replace(self, atoms=new_atoms)
