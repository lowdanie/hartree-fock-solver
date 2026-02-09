import numpy as np

from tests.jax_utils import pytree_utils

import slaterform as sf


def test_atom_pytree():
    atom = sf.Atom(
        symbol="H",
        number=1,
        position=np.array([0.0, 0.0, 0.0]),
        shells=[
            sf.ContractedGTO(
                primitive_type=sf.PrimitiveType.CARTESIAN,
                angular_momentum=(0,),
                exponents=np.array([1.0], dtype=np.float64),
                coefficients=np.array([[2.0]], dtype=np.float64),
            )
        ],
    )
    pytree_utils.assert_valid_pytree(atom)


def test_with_position():
    atom = sf.Atom(
        symbol="H",
        number=1,
        position=np.array([0.0, 0.0, 0.0]),
        shells=[
            sf.ContractedGTO(
                primitive_type=sf.PrimitiveType.CARTESIAN,
                angular_momentum=(0,),
                exponents=np.array([1.0], dtype=np.float64),
                coefficients=np.array([[2.0]], dtype=np.float64),
            )
        ],
    )

    new_position = np.array([1.0, 1.0, 1.0])
    new_atom = atom.with_position(new_position)

    np.testing.assert_array_equal(new_atom.position, new_position)
    assert new_atom.symbol == atom.symbol
    assert new_atom.number == atom.number

    for shell1, shell2 in zip(new_atom.shells, atom.shells):
        assert shell1 is shell2
