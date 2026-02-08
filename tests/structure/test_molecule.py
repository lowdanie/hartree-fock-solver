from unittest import mock
import numpy as np

from tests.jax_utils import pytree_utils
import slaterform as sf

_DUMMY_SHELL_H = "Shell_H"
_DUMMY_SHELL_O = "Shell_O"

_TEST_SHELL_H = sf.ContractedGTO(
    primitive_type=sf.PrimitiveType.CARTESIAN,
    angular_momentum=(0,),
    exponents=np.array([1.0]),
    coefficients=np.array([2.0]),
)

_TEST_SHELL_O = sf.ContractedGTO(
    primitive_type=sf.PrimitiveType.CARTESIAN,
    angular_momentum=(1,),
    exponents=np.array([3.0]),
    coefficients=np.array([4.0]),
)

_TEST_MOLECULE_HO = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0]),
            shells=[_TEST_SHELL_H],
        ),
        sf.Atom(
            symbol="O",
            number=8,
            position=np.array([1.0, 0.0, 0.0]),
            shells=[_TEST_SHELL_O],
        ),
    ]
)


def assert_shell_equal(shell1: sf.ContractedGTO, shell2: sf.ContractedGTO):
    """Asserts that two ContractedGTO shells are equal."""
    assert (
        shell1.primitive_type == shell2.primitive_type
    ), "Primitive types differ"
    assert (
        shell1.angular_momentum == shell2.angular_momentum
    ), "Angular momenta differ"
    np.testing.assert_array_equal(
        shell1.exponents,
        shell2.exponents,
        err_msg="Exponents differ",
    )
    np.testing.assert_array_equal(
        shell1.coefficients,
        shell2.coefficients,
        err_msg="Coefficients differ",
    )


def assert_atom_equal(atom1: sf.Atom, atom2: sf.Atom):
    """Asserts that two Atom objects are equal."""
    assert (
        atom1.symbol == atom2.symbol
    ), f"Atom symbols differ: {atom1.symbol} vs {atom2.symbol}"
    assert (
        atom1.number == atom2.number
    ), f"Atomic numbers differ: {atom1.number} vs {atom2.number}"
    np.testing.assert_array_equal(
        atom1.position,
        atom2.position,
        err_msg=f"Atom positions differ for {atom1.symbol}",
    )
    assert len(atom1.shells) == len(
        atom2.shells
    ), f"Number of shells differ for {atom1.symbol}"
    for shell1, shell2 in zip(atom1.shells, atom2.shells):
        assert_shell_equal(shell1, shell2)


def assert_molecule_equal(mol1: sf.Molecule, mol2: sf.Molecule):
    """Asserts that two Molecule objects are equal."""
    assert len(mol1.atoms) == len(mol2.atoms), "Number of atoms differ"
    for atom1, atom2 in zip(mol1.atoms, mol2.atoms):
        assert_atom_equal(atom1, atom2)


def _mock_bse_load_side_effect(basis_name, element):
    """Simulates fetching different shells for H (1) and O (8)."""
    if basis_name != "sto-3g":
        raise ValueError("Unexpected basis name")

    if element == 1:
        return [_DUMMY_SHELL_H]
    elif element == 8:
        return [_DUMMY_SHELL_O]
    else:
        raise ValueError(f"Unknown element: {element}")


def test_molecule_pytree():
    pytree_utils.assert_valid_pytree(_TEST_MOLECULE_HO)


@mock.patch(
    "slaterform.structure.molecule.load_basis",
    side_effect=_mock_bse_load_side_effect,
)
def test_from_geometry(mock_bse_load):
    geometry = [
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0]),
            shells=[],
        ),
        sf.Atom(
            symbol="O",
            number=8,
            position=np.array([1.0, 0.0, 0.0]),
            shells=[],
        ),
    ]

    molecule = sf.Molecule.from_geometry(geometry, basis_name="sto-3g")

    assert len(molecule.atoms) == 2
    assert molecule.atoms[0].symbol == "H"
    assert molecule.atoms[0].number == 1
    np.testing.assert_array_equal(
        molecule.atoms[0].position, np.array([0.0, 0.0, 0.0])
    )
    assert molecule.atoms[0].shells == [_DUMMY_SHELL_H]
    assert molecule.atoms[1].symbol == "O"
    assert molecule.atoms[1].number == 8
    np.testing.assert_array_equal(
        molecule.atoms[1].position, np.array([1.0, 0.0, 0.0])
    )
    assert molecule.atoms[1].shells == [_DUMMY_SHELL_O]


def test_update_positions():
    new_positions = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    expected_molecule = sf.Molecule(
        atoms=[
            sf.Atom(
                symbol="H",
                number=1,
                position=new_positions[0],
                shells=[_TEST_SHELL_H],
            ),
            sf.Atom(
                symbol="O",
                number=8,
                position=new_positions[1],
                shells=[_TEST_SHELL_O],
            ),
        ]
    )
    updated_molecule = _TEST_MOLECULE_HO.with_positions(new_positions)
    assert_molecule_equal(updated_molecule, expected_molecule)


def test_n_basis():
    assert _TEST_MOLECULE_HO.n_basis == 4
