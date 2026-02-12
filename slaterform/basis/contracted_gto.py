import dataclasses
import enum

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import slaterform.types as types
from slaterform.basis.cartesian import generate_cartesian_powers


class PrimitiveType(enum.Enum):
    """The type of primitive Gaussian function."""

    CARTESIAN = 1
    SPHERICAL = 2


@register_pytree_node_class
@dataclasses.dataclass
class ContractedGTO:
    """A contracted Gaussian-type orbital (Contracted GTO).

    This represents a family of contracted Gaussian basis functions centered at
    a given point in 3D space.

    A contracted Gaussian basis function is defined as a linear combination of
    normalized primitive Gaussian functions with the same center but different
    exponents:

    psi(r) = sum_{d=1}^K c_d * N(alpha_d, l, m) Y(r; l, m) e^(-alpha_d * |r - A|^2)

    where:
    1. A = (A_x, A_y, A_z) is the center
    2. r = (x, y, z) is the position coordinate
    3. alpha_d = exponents[d] for 0 <= d < K
    4. c_d = coefficients[s, d] for some 0 <= s < num_shells
    5. Y(r; l, m) is an (unnormalized) angular function with momentum
       l = angular_momentum[s]
       in either Cartesian or spherical representation, depending on primitive_type.
    6. N(alpha, l, m) is the inverse of the L^2 norm of
       Y(r; l, m) e^(-alpha_d * |r - A|^2)
    """

    primitive_type: PrimitiveType

    # The angular momentum for each shell in this contracted GTO.
    # shape (N_shell,)
    angular_momentum: tuple[int, ...]

    # The exponents for the Gaussian primitives in this contracted GTO.
    # shape (K,)
    exponents: types.Array

    # The contraction coefficients for each shell in this contracted GTO.
    # shape (N_shell, K)
    coefficients: types.Array

    def __post_init__(self):
        types.promote_dataclass_fields(self)
        self.exponents = types.safe_cast(self.exponents, jnp.float64)
        self.coefficients = types.safe_cast(self.coefficients, jnp.float64)

    def tree_flatten(self):
        children = (self.exponents, self.coefficients)
        aux_data = (self.primitive_type, self.angular_momentum)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[PrimitiveType, tuple[int, ...]],
        children: tuple[jax.Array, jax.Array],
    ) -> "ContractedGTO":
        return cls(
            primitive_type=aux_data[0],
            angular_momentum=aux_data[1],
            exponents=children[0],
            coefficients=children[1],
        )

    @property
    def n_basis(self) -> int:
        if self.primitive_type != PrimitiveType.CARTESIAN:
            raise NotImplementedError(
                "Only Cartesian contracted GTOs are supported currently."
            )

        return sum(
            len(generate_cartesian_powers(l)) for l in self.angular_momentum
        )
