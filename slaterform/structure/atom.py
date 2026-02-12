import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import slaterform.types as types
from slaterform.basis.contracted_gto import ContractedGTO


@register_pytree_node_class
@dataclasses.dataclass
class Atom:
    symbol: str
    number: int  # Atomic number

    # Position in Bohr units
    position: types.Array

    # The basis functions centered on this atom.
    shells: Sequence[ContractedGTO] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        types.promote_dataclass_fields(self)
        self.position = types.safe_cast(self.position, jnp.float64)

    def tree_flatten(self):
        children = (self.position, self.shells)
        aux_data = (self.symbol, self.number)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[str, int],
        children: tuple[jax.Array, Sequence[ContractedGTO]],
    ) -> "Atom":
        return cls(
            symbol=aux_data[0],
            number=aux_data[1],
            position=children[0],
            shells=children[1],
        )

    def with_position(self, new_position: types.Array) -> "Atom":
        """Returns a new Atom with an updated position."""
        return dataclasses.replace(self, position=new_position)
