import pytest
import itertools

import jax
from jax import jit
from jax import numpy as jnp
import numpy as np

import pubchempy as pcp

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


def _build_h2_positions(bond_length: jax.Array) -> jax.Array:
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, bond_length],
        ],
        dtype=jnp.float64,
    )


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


def _logger(status: scf.SolverStatus):
    print(f"Iteration {status.iteration}: err = {status.err:.8f}")


_SOLVERS = [
    scf.LinearMixingParams(max_iter=20, static_loop=True),
    scf.LinearMixingParams(max_iter=20, static_loop=False),
    scf.AndersonParams(max_iter=20, static_loop=True),
    scf.AndersonParams(max_iter=20, static_loop=False),
]

_INTEGRAL_STRATEGIES = [scf.DirectStrategy(), scf.CachedStrategy()]

_IMPLICIT_DIFF_OPTIONS = [False, True]


def _generate_options():
    perturbation = 0.0
    for solver, integral_strategy, implicit_diff in itertools.product(
        _SOLVERS, _INTEGRAL_STRATEGIES, _IMPLICIT_DIFF_OPTIONS
    ):
        yield scf.Options(
            solver, integral_strategy, perturbation, implicit_diff
        )


def _generate_options_with_gradients():
    perturbation = 1e-10
    for solver, integral_strategy, implicit_diff in itertools.product(
        _SOLVERS, _INTEGRAL_STRATEGIES, _IMPLICIT_DIFF_OPTIONS
    ):
        if solver.static_loop or implicit_diff:
            yield scf.Options(
                solver, integral_strategy, perturbation, implicit_diff
            )


@pytest.mark.parametrize(
    "options",
    list(_generate_options()),
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


def test_H2_initial_density():
    solve_fn = jit(scf.solve)
    basis = sf.BatchedBasis.from_molecule(_H2_MOLECULE)
    result = solve_fn(basis)

    new_result = solve_fn(basis, P0=result.density)
    np.testing.assert_almost_equal(
        new_result.electronic_energy,
        _EXPECTED_ELECTRONIC_ENERGY_H2,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        new_result.total_energy,
        _EXPECTED_TOTAL_ENERGY_H2,
        decimal=4,
    )


def test_H2_integrals_f32():
    solve_fn = jit(scf.solve)
    basis = sf.BatchedBasis.from_molecule(
        _H2_MOLECULE, batch_size_1e=2, batch_size_2e=2
    )
    options = scf.Options(
        solver=sf.fixed_point.LinearMixingParams(),
        integral_strategy=scf.CachedStrategy(dtype=jnp.float32),
        perturbation=1e-10,
        implicit_diff=False,
    )

    result = solve_fn(basis, options)

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


def test_H2_integrals_compute_f32():
    solve_fn = jit(scf.solve)
    basis = sf.BatchedBasis.from_molecule(
        _H2_MOLECULE, batch_size_1e=2, batch_size_2e=2
    )
    options = scf.Options(
        solver=sf.fixed_point.LinearMixingParams(),
        integral_strategy=scf.CachedStrategy(
            dtype=jnp.float64, compute_dtype=jnp.float32
        ),
        perturbation=1e-10,
        implicit_diff=False,
    )

    result = solve_fn(basis, options)

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


def test_H2_compile_only():
    lowered = jit(scf.solve).lower(_H2_MOLECULE)
    lowered.compile()


@pytest.mark.parametrize(
    "options",
    list(_generate_options_with_gradients()),
)
def test_H2_gradients(options: scf.Options):
    molecule = _H2_MOLECULE
    basis = sf.BatchedBasis.from_molecule(
        molecule, batch_size_1e=2, batch_size_2e=4
    )

    def energy(bond_length: jax.Array) -> jax.Array:
        new_positions = _build_h2_positions(bond_length)
        new_basis = basis.with_positions(new_positions)
        result = scf.solve(new_basis, options)
        return result.total_energy

    val_and_grad_fn = jit(jax.value_and_grad(energy))
    E, grad_E = val_and_grad_fn(1.4)

    np.testing.assert_almost_equal(E, _EXPECTED_TOTAL_ENERGY_H2, decimal=4)
    assert not np.isnan(grad_E)
    assert np.abs(grad_E) < 0.1


def test_H2_implicit_grad_consistency():
    bond_length = 1.4
    molecule = _H2_MOLECULE
    basis = sf.BatchedBasis.from_molecule(molecule, batch_size_2e=2)

    def energy_fixed(r):
        new_positions = _build_h2_positions(r)
        new_basis = basis.with_positions(new_positions)
        options = scf.Options(
            solver=sf.fixed_point.LinearMixingParams(
                static_loop=True,
            ),
            perturbation=1e-10,
            implicit_diff=False,
        )
        return scf.solve(new_basis, options).total_energy

    grad_fixed = jit(jax.grad(energy_fixed))(bond_length)

    def energy_implicit(r):
        new_positions = _build_h2_positions(r)
        new_basis = basis.with_positions(new_positions)
        options = scf.Options(
            solver=sf.fixed_point.LinearMixingParams(
                static_loop=False,
            ),
            perturbation=1e-10,
            implicit_diff=True,
        )
        return scf.solve(new_basis, options).total_energy

    grad_implicit = jit(jax.grad(energy_implicit))(bond_length)

    # 3. Assert they are compatible
    print(f"Gradient Fixed:    {grad_fixed:.8f}")
    print(f"Gradient Implicit: {grad_implicit:.8f}")

    np.testing.assert_allclose(grad_fixed, grad_implicit, rtol=1e-4, atol=1e-4)


def test_build_initial_density():
    expected = np.zeros((2, 2), dtype=np.float64)
    density = scf.build_initial_density(_H2_MOLECULE)

    np.testing.assert_equal(density, expected)


@pytest.mark.slow
@pytest.mark.parametrize(
    "options",
    [
        scf.Options(
            solver=sf.fixed_point.LinearMixingParams(
                max_iter=50, damping=0.1, static_loop=False, callback=_logger
            ),
        ),
        scf.Options(
            solver=sf.fixed_point.AndersonParams(
                max_iter=50, m=5, beta=0.9, static_loop=False, callback=_logger
            ),
        ),
    ],
)
def test_H2O(options: scf.Options):
    basis = sf.BatchedBasis.from_molecule(_H2O_MOLECULE)
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
        mol = _H2O_MOLECULE.with_positions(positions)

        options = scf.Options(
            solver=sf.fixed_point.LinearMixingParams(
                max_iter=20, static_loop=True, callback=_logger
            ),
            perturbation=1e-10,
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


@pytest.mark.slow
def test_aspirin_memory_analysis():
    batch_size = 1024
    compute_grad = True
    print("Experiment:")
    print("Molecule: Aspirin")
    print("Batch size 2e: ", batch_size)
    print("With grad: ", compute_grad)

    compounds = pcp.get_compounds("Aspirin", "name", record_type="3d")
    atoms = sf.adapters.pubchem.load_geometry(compounds[0])
    molecule = sf.Molecule.from_geometry(atoms, basis_name="sto-3g")

    def total_energy(positions: jax.Array):
        mol = molecule.with_positions(positions)
        basis = sf.BatchedBasis.from_molecule(mol, batch_size_2e=batch_size)
        options = scf.Options(
            solver=sf.fixed_point.LinearMixingParams(
                max_iter=20, static_loop=True
            ),
            perturbation=1e-10,
        )
        result = scf.solve(basis, options)

        return result.total_energy

    positions = jnp.array([atom.position for atom in molecule.atoms])

    print("Running...")
    if compute_grad:
        value_and_grad_fn = jit(jax.value_and_grad(total_energy))
    else:
        value_and_grad_fn = jit(total_energy)

    lowered = value_and_grad_fn.lower(positions)
    compiled = lowered.compile()
    mem_analysis = compiled.memory_analysis()

    print(f"Temp Heap Size: {mem_analysis.temp_size_in_bytes / 1024**2:.2f} MB")
