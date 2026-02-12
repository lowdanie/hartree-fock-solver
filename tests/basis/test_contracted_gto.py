import numpy as np
import jax.numpy as jnp

import slaterform as sf
from tests.jax_utils import pytree_utils

_TEST_CGTO = sf.ContractedGTO(
    primitive_type=sf.PrimitiveType.CARTESIAN,
    angular_momentum=(0, 1),
    exponents=np.array([0.1, 0.2], dtype=np.float64),
    coefficients=np.array([[1.0, 0.5], [0.3, 0.2]], dtype=np.float64),
)


def test_contracted_gto_pytree():
    pytree_utils.assert_valid_pytree(_TEST_CGTO)


def test_float64_dtype():
    gto = sf.ContractedGTO(
        primitive_type=sf.PrimitiveType.CARTESIAN,
        angular_momentum=(0, 1),
        exponents=np.array([0.1, 0.2], dtype=np.float32),
        coefficients=np.array([[1.0, 0.5], [0.3, 0.2]], dtype=np.float32),
    )

    assert gto.exponents.dtype == jnp.float64
    assert gto.coefficients.dtype == jnp.float64


def test_n_basis():
    assert _TEST_CGTO.n_basis == 4
