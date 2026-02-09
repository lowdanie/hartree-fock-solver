import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

import slaterform.types as types
from slaterform.basis.contracted_gto import ContractedGTO, PrimitiveType
from slaterform.basis.cartesian import (
    compute_normalization_constants,
    generate_cartesian_powers,
)


def _compute_normalization_factors(
    max_degree: int, cartesian_powers: types.StaticArray, exponents: types.Array
) -> jax.Array:
    f = functools.partial(
        compute_normalization_constants, max_degree, cartesian_powers
    )
    return jax.vmap(f)(exponents).T


@register_pytree_node_class
@dataclasses.dataclass
class BasisBlock:
    """A block representation of a contracted Gaussian-type orbital basis.

    This flattens the angular momentum shells into their individual Cartesian
    components and pre-multiplies them by normalization constants. It also
    provides a basis transformation matrix from the Cartesian representation to
    the contracted basis function representation (Cartesian or spherical).

    Note that N_cart is equal to the total number of Cartesian basis
    functions across all shells, i.e.
    N_cart = sum_{l=0}^{num_shells - 1} (angular_momentum[l] + 2 choose 2)
    """

    # shape (3,)
    center: types.Array

    # The common set of exponents for the Gaussian primitives in this block.
    # shape (K,)
    exponents: types.Array

    # The powers (i,j,k) for each Cartesian basis function in this block.
    # shape (N_cart, 3)
    cartesian_powers: types.StaticArray

    # A map from the Gaussian primitives to their normalized contraction
    # coefficients for each Cartesian basis function in this block.
    # shape (N_cart, K)
    contraction_matrix: types.Array

    # A map from the Cartesian basis functions to the angular basis functions
    # in this block (Cartesian or spherical).
    # shape (N_basis, N_cart)
    basis_transform: types.Array

    @property
    def n_exponents(self) -> int:
        """The number of Gaussian primitives in this block."""
        return self.exponents.shape[0]

    @property
    def n_cart(self) -> int:
        """The number of Cartesian basis functions in this block."""
        return self.cartesian_powers.shape[0]

    @property
    def n_basis(self) -> int:
        """The number of basis functions in this block."""
        return self.basis_transform.shape[0]

    @property
    def max_degree(self) -> int:
        """The maximum angular momentum degree in this block."""
        return int(np.max(self.cartesian_powers))

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (
            self.center,
            self.exponents,
            self.contraction_matrix,
            self.basis_transform,
        )
        aux_data = tuple(map(tuple, self.cartesian_powers.tolist()))
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: int, children: tuple[jax.Array, ...]
    ) -> "BasisBlock":
        return cls(
            center=children[0],
            exponents=children[1],
            cartesian_powers=np.array(aux_data, dtype=np.int32),
            contraction_matrix=children[2],
            basis_transform=children[3],
        )

    @classmethod
    def from_gto(cls, gto: ContractedGTO, center: types.Array) -> "BasisBlock":
        """Builds a BasisBlock from a ContractedGTO at the given center."""
        if gto.primitive_type != PrimitiveType.CARTESIAN:
            raise NotImplementedError(
                "Only Cartesian contracted GTOs are supported currently."
            )
        max_degree = max(gto.angular_momentum)
        power_blocks = [
            generate_cartesian_powers(l) for l in gto.angular_momentum
        ]
        power_block_sizes = np.array(
            [powers.shape[0] for powers in power_blocks]
        )

        # The cartesian powers are static and can be stored as a numpy array.
        cartesian_powers = np.vstack(power_blocks)

        contraction_matrix = jnp.repeat(
            gto.coefficients, power_block_sizes, axis=0
        )
        norm_factors = _compute_normalization_factors(
            max_degree,
            cartesian_powers,
            gto.exponents,
        )
        basis_transform = jnp.eye(
            cartesian_powers.shape[0], dtype=contraction_matrix.dtype
        )

        return cls(
            center=center,
            exponents=gto.exponents,
            cartesian_powers=cartesian_powers,
            contraction_matrix=norm_factors * contraction_matrix,
            basis_transform=basis_transform,
        )

    def with_center(self, new_center: types.Array) -> "BasisBlock":
        """Returns a new BasisBlock with an updated center."""
        return dataclasses.replace(self, center=new_center)
