import dataclasses
import pytest

from jax import jit
import numpy as np

import slaterform as sf


@dataclasses.dataclass
class _BuildBasisBlockTestCase:
    gto: sf.ContractedGTO
    center: np.ndarray  # shape (3,)
    expected: sf.BasisBlock


# The contraction coefficients were normalized using the following code:
# import numpy as np
# import sympy as sp
#
#
# x, y, z = sp.symbols('x,y,z')
# a = 0.1
# i,j,k = 0,0,0
# coeff = 0.5
#
# integrand = (x**i * y**j * z**k * sp.exp(-a *(x**2 + y**2 + z**2)))**2
# res = sp.integrate(integrand, (x,-sp.oo,sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
# norm_sq = float(res.evalf())
# inv_norm = 1 / np.sqrt(norm_sq)
# print(f"normalized coeff: {inv_norm * coeff:.8f}")
@pytest.mark.parametrize(
    "case",
    [
        _BuildBasisBlockTestCase(
            gto=sf.ContractedGTO(
                primitive_type=sf.PrimitiveType.CARTESIAN,
                angular_momentum=(0, 0, 1),
                exponents=np.array([0.1, 0.2, 0.3], dtype=np.float64),
                coefficients=np.array(
                    [
                        [0.5, 0.4, 0.3],
                        [0.4, 0.3, 0.2],
                        [0.6, 0.5, 0.4],
                    ],
                    dtype=np.float64,
                ),
            ),
            center=np.array([1.0, 2.0, 3.0]),
            expected=sf.BasisBlock(
                center=np.array([1.0, 2.0, 3.0]),
                exponents=np.array([0.1, 0.2, 0.3], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.06336947, 0.08525946, 0.08667070],
                        [0.05069558, 0.06394460, 0.05778046],
                        [0.04809405, 0.09532297, 0.12659066],
                        [0.04809405, 0.09532297, 0.12659066],
                        [0.04809405, 0.09532297, 0.12659066],
                    ],
                ),
                basis_transform=np.eye(5, dtype=np.float64),
            ),
        ),
    ],
)
def test_build_basis_block(case):
    result = jit(sf.BasisBlock.from_gto)(case.gto, case.center)
    assert result.n_basis == case.expected.n_basis
    assert result.n_cart == case.expected.n_cart

    np.testing.assert_allclose(
        result.center,
        case.expected.center,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        result.exponents,
        case.expected.exponents,
        rtol=1e-7,
    )
    np.testing.assert_array_equal(
        result.cartesian_powers,
        case.expected.cartesian_powers,
    )
    np.testing.assert_allclose(
        result.contraction_matrix,
        case.expected.contraction_matrix,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        result.basis_transform,
        case.expected.basis_transform,
        rtol=1e-7,
    )


def test_build_basis_block_unsupported_primitive():
    gto = sf.ContractedGTO(
        primitive_type=sf.PrimitiveType.SPHERICAL,
        angular_momentum=(0,),
        exponents=np.array([0.1], dtype=np.float64),
        coefficients=np.array([[1.0]], dtype=np.float64),
    )
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(NotImplementedError):
        sf.BasisBlock.from_gto(gto, center)


def test_n_basis():
    block = sf.BasisBlock(
        center=np.array([0.0, 0.0, 0.0]),
        exponents=np.array([0.1], dtype=np.float64),
        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        contraction_matrix=np.array([[1.0], [0.5], [0.1]], dtype=np.float64),
        basis_transform=np.ones((2, 3), dtype=np.float64),
    )

    assert block.n_basis == 2


def test_n_cart():
    block = sf.BasisBlock(
        center=np.array([0.0, 0.0, 0.0]),
        exponents=np.array([0.1], dtype=np.float64),
        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        contraction_matrix=np.array([[1.0], [0.5], [0.1]], dtype=np.float64),
        basis_transform=np.ones((2, 3), dtype=np.float64),
    )

    assert block.n_cart == 3


def test_max_degree():
    block = sf.BasisBlock(
        center=np.array([0.0, 0.0, 0.0]),
        exponents=np.array([0.1], dtype=np.float64),
        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0], [0, 2, 1]]),
        contraction_matrix=np.array([[1.0], [0.5], [0.1]], dtype=np.float64),
        basis_transform=np.ones((2, 3), dtype=np.float64),
    )

    assert block.max_degree == 2


def test_with_center():
    block = sf.BasisBlock(
        center=np.array([0.0, 0.0, 0.0]),
        exponents=np.array([0.1], dtype=np.float64),
        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        contraction_matrix=np.array([[1.0], [0.5], [0.1]], dtype=np.float64),
        basis_transform=np.ones((2, 3), dtype=np.float64),
    )

    new_center = np.array([1.0, 1.0, 1.0])
    new_block = block.with_center(new_center)

    np.testing.assert_array_equal(new_block.center, new_center)
    assert new_block.exponents is block.exponents
    assert new_block.cartesian_powers is block.cartesian_powers
    assert new_block.contraction_matrix is block.contraction_matrix
    assert new_block.basis_transform is block.basis_transform
