import pytest

import jax
from jax import jit
from jax import numpy as jnp
import numpy as np

import slaterform as sf
import slaterform.hartree_fock.scf as scf
from tests.jax_utils import pytree_utils

_H_SHELLS = sf.adapters.bse.load("sto-3g", 1)
_O_SHELLS = sf.adapters.bse.load("sto-3g", 8)

_H2_MOLECULE = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            shells=_H_SHELLS,
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 1.4], dtype=np.float64),
            shells=_H_SHELLS,
        ),
    ]
)

# The standard STO-3G basis set energies for H2 at
# 1.4 Bohr bond length.
_EXPECTED_ELECTRONIC_ENERGY_H2 = -1.8310  # Hartree
_EXPECTED_TOTAL_ENERGY_H2 = -1.1167  # Hartree

# Water molecule in Bohr units. The geometry is from pubchem.
_H2O_MOLECULE = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="O",
            number=8,
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            shells=_O_SHELLS,
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array(
                [0.52421003, 1.68733646, 0.48074633], dtype=np.float64
            ),
            shells=_H_SHELLS,
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array(
                [1.14668581, -0.45032174, -1.35474466], dtype=np.float64
            ),
            shells=_H_SHELLS,
        ),
    ]
)


# The electronic energy for H2O with the STO-3G basis set.
# Computed with PySCF for reference.
# import numpy as np
# from pyscf import gto, scf
#
# mol = gto.M(
#     atom = '''
#     O 0.000000000  0.000000000 0.000000000
#     H 0.52421003 1.68733646 0.48074633
#     H 1.14668581, -0.45032174, -1.35474466
#     ''',
#     basis = 'sto-3g',
#     unit = 'Bohr',
#     symmetry = False
# )
#
# mf = scf.RHF(mol)
# mf.verbose = 4
# mf.kernel()
#
# print(f"Electronic Energy: {mf.e_tot - mol.energy_nuc():.8f} Ha")
# print(f"Total Energy:      {mf.e_tot:.4f} Ha")
_EXPECTED_ELECTRONIC_ENERGY_H2O = -84.04881208  # Hartree
_EXPECTED_TOTAL_ENERGY_H2O = -74.96444758  # Hartree


def _build_h2_molecule(bond_length: jax.Array) -> sf.Molecule:
    return sf.Molecule(
        atoms=[
            sf.Atom(
                symbol="H",
                number=1,
                position=jnp.array([0.0, 0.0, 0.0], dtype=np.float64),
                shells=_H_SHELLS,
            ),
            sf.Atom(
                symbol="H",
                number=1,
                position=jnp.array([0.0, 0.0, bond_length], dtype=np.float64),
                shells=_H_SHELLS,
            ),
        ]
    )


def _update_molecule_geometry(
    template_mol: sf.Molecule, new_positions: jax.Array
) -> sf.Molecule:
    """Creates a new Molecule with the new positions."""
    new_atoms = []
    for i, atom in enumerate(template_mol.atoms):
        new_atoms.append(
            sf.Atom(
                symbol=atom.symbol,
                number=atom.number,
                position=new_positions[i],
                shells=atom.shells,
            )
        )
    return sf.Molecule(new_atoms)


def test_callback_options_pytree():
    options = scf.CallbackOptions(
        interval=10,
        func=lambda state: print("test callback"),
    )

    pytree_utils.assert_valid_pytree(options)


def test_callback_options_zero_callback_intervals():
    with pytest.raises(ValueError):
        scf.CallbackOptions(interval=0)


def test_options_pytree():
    pytree_utils.assert_valid_pytree(scf.Options())


def test_context_pytree():
    context = scf.Context(
        basis=sf.BatchedBasis.from_molecule(_H2_MOLECULE),
        nuclear_energy=jnp.asarray(1.0),
        S=jnp.ones((2, 2)),
        X=2 * jnp.ones((2, 2)),
        H_core=3 * jnp.ones((2, 2)),
        V=jnp.ones((2, 2, 2, 2)),
    )

    pytree_utils.assert_valid_pytree(context)


def test_state_pytree():
    state = scf.State(
        C=4 * jnp.ones((2, 2)),
        P=5 * jnp.ones((2, 2)),
        F=6 * jnp.ones((2, 2)),
        electronic_energy=7 * jnp.asarray(1.0),
        total_energy=8 * jnp.asarray(1.0),
        orbital_energies=9 * jnp.ones((2,)),
        delta_P_sq=10 * jnp.asarray(1.0),
    )

    pytree_utils.assert_valid_pytree(state)


def test_result_pytree():
    basis = sf.BatchedBasis.from_molecule(_H2_MOLECULE)
    result = scf.Result(
        converged=jnp.asarray(True),
        iterations=jnp.array(10, dtype=jnp.int32),
        electronic_energy=jnp.asarray(-1.0),
        nuclear_energy=jnp.asarray(1.0),
        total_energy=jnp.asarray(0.0),
        basis=basis,
        orbital_energies=jnp.ones((2,)),
        orbitals=jnp.ones((2, 2)),
        density=jnp.ones((2, 2)),
    )

    pytree_utils.assert_valid_pytree(result)


@pytest.mark.parametrize(
    "options",
    [
        scf.Options(
            execution_mode=scf.ExecutionMode.CONVERGENCE,
            integral_strategy=scf.IntegralStrategy.DIRECT,
        ),
        scf.Options(
            max_iterations=10,
            execution_mode=scf.ExecutionMode.FIXED,
            integral_strategy=scf.IntegralStrategy.DIRECT,
        ),
        scf.Options(
            execution_mode=scf.ExecutionMode.IMPLICIT,
            integral_strategy=scf.IntegralStrategy.DIRECT,
        ),
        scf.Options(
            execution_mode=scf.ExecutionMode.CONVERGENCE,
            integral_strategy=scf.IntegralStrategy.CACHED,
        ),
        scf.Options(
            max_iterations=10,
            execution_mode=scf.ExecutionMode.FIXED,
            integral_strategy=scf.IntegralStrategy.CACHED,
        ),
        scf.Options(
            execution_mode=scf.ExecutionMode.IMPLICIT,
            integral_strategy=scf.IntegralStrategy.CACHED,
        ),
        scf.Options.differentiable(steps=10),
    ],
)
def test_H2(options: scf.Options):
    basis = sf.BatchedBasis.from_molecule(_H2_MOLECULE)
    result = jit(scf.solve)(basis, options)

    np.testing.assert_almost_equal(
        result.electronic_energy,
        _EXPECTED_ELECTRONIC_ENERGY_H2,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        result.total_energy,
        _EXPECTED_TOTAL_ENERGY_H2,
        decimal=4,
    )


def test_H2_from_molecule():
    result = jit(scf.solve)(_H2_MOLECULE)

    np.testing.assert_almost_equal(
        result.electronic_energy,
        _EXPECTED_ELECTRONIC_ENERGY_H2,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        result.total_energy,
        _EXPECTED_TOTAL_ENERGY_H2,
        decimal=4,
    )


def test_H2_gradients():
    def energy(bond_length: jax.Array) -> jax.Array:
        mol = _build_h2_molecule(bond_length)
        options = scf.Options.differentiable(
            steps=20,
            callback=scf.CallbackOptions(
                interval=1,
                func=lambda step: print(
                    f"Iteration {step.iteration}: E = {step.state.electronic_energy:.8f} Ha"
                ),
            ),
        )
        result = scf.solve(mol, options)
        return result.total_energy

    val_and_grad_fn = jit(jax.value_and_grad(energy))
    E, grad_E = val_and_grad_fn(1.4)

    np.testing.assert_almost_equal(E, _EXPECTED_TOTAL_ENERGY_H2, decimal=4)
    assert not np.isnan(grad_E)
    assert np.abs(grad_E) < 0.1


def test_H2_implicit_gradients():
    def energy(bond_length: jax.Array) -> jax.Array:
        mol = _build_h2_molecule(bond_length)
        options = scf.Options(
            execution_mode=scf.ExecutionMode.IMPLICIT,
            integral_strategy=scf.IntegralStrategy.CACHED,
        )
        result = scf.solve(mol, options)
        return result.total_energy

    val_and_grad_fn = jit(jax.value_and_grad(energy))
    E, grad_E = val_and_grad_fn(1.4)

    np.testing.assert_almost_equal(E, _EXPECTED_TOTAL_ENERGY_H2, decimal=4)
    assert not np.isnan(grad_E)
    assert np.abs(grad_E) < 0.1


def test_H2_implicit_grad_consistency():
    bond_length = 1.4

    def energy_fixed(r):
        mol = _build_h2_molecule(r)
        options = scf.Options.differentiable(steps=50)
        return scf.solve(mol, options).total_energy

    grad_fixed = jit(jax.grad(energy_fixed))(bond_length)

    def energy_implicit(r):
        mol = _build_h2_molecule(r)
        options = scf.Options(
            execution_mode=scf.ExecutionMode.IMPLICIT,
            integral_strategy=scf.IntegralStrategy.CACHED,
            convergence_threshold=1e-8,
        )
        return scf.solve(mol, options).total_energy

    grad_implicit = jit(jax.grad(energy_implicit))(bond_length)

    # 3. Assert they are compatible
    print(f"\nGradient Fixed:    {grad_fixed:.8f}")
    print(f"Gradient Implicit: {grad_implicit:.8f}")

    np.testing.assert_allclose(grad_fixed, grad_implicit, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
def test_H2O():
    basis = sf.BatchedBasis.from_molecule(_H2O_MOLECULE)
    options = scf.Options(
        max_iterations=20,
        execution_mode=scf.ExecutionMode.FIXED,
        integral_strategy=scf.IntegralStrategy.CACHED,
        callback=scf.CallbackOptions(
            interval=1,
            func=lambda step: print(
                f"Iteration {step.iteration}: E = {step.state.electronic_energy:.8f} Ha"
            ),
        ),
    )

    result = jit(scf.solve)(basis, options)

    np.testing.assert_almost_equal(
        result.electronic_energy,
        _EXPECTED_ELECTRONIC_ENERGY_H2O,
        decimal=5,
    )
    np.testing.assert_almost_equal(
        result.total_energy,
        _EXPECTED_TOTAL_ENERGY_H2O,
        decimal=5,
    )


@pytest.mark.slow
def test_H2O_grad():
    def total_energy(positions: jax.Array):
        mol = _update_molecule_geometry(_H2O_MOLECULE, positions)

        options = scf.Options(
            max_iterations=20,
            execution_mode=scf.ExecutionMode.FIXED,
            integral_strategy=scf.IntegralStrategy.CACHED,
            perturbation=1e-10,
            convergence_threshold=1e-6,
        )
        result = scf.solve(mol, options)

        return result.total_energy

    total_energy_and_grad = jit(jax.value_and_grad(total_energy))
    positions = jnp.array([atom.position for atom in _H2O_MOLECULE.atoms])

    energy, _ = total_energy_and_grad(positions)

    np.testing.assert_almost_equal(
        energy,
        _EXPECTED_TOTAL_ENERGY_H2O,
        decimal=5,
    )
