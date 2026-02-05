import dataclasses
from typing import Callable, NamedTuple, Optional
import functools
import enum

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


import slaterform.types as types

from slaterform.fixed_point.fixed_point import SolverStatus
from slaterform.fixed_point.anderson import AndersonParams
from slaterform.fixed_point.anderson import solve as anderson_solve
from slaterform.fixed_point.linear_mixing import LinearMixingParams
from slaterform.fixed_point.linear_mixing import solve as lm_solve

from slaterform.hartree_fock.density import closed_shell_matrix
from slaterform.hartree_fock.fock import (
    two_electron_matrix,
    two_electron_integrals,
    two_electron_matrix_from_integrals,
    electronic_energy,
)
from slaterform.hartree_fock.one_electron import (
    core_hamiltonian_matrix,
    overlap_matrix,
)
from slaterform.hartree_fock.roothaan import (
    orthogonalize_basis,
    solve as solve_roothaan,
)
from slaterform.fixed_point.implicit import attach_implicit_grad
from slaterform.structure.batched_basis import BatchedBasis
from slaterform.structure.molecule import Molecule
from slaterform.structure.nuclear import (
    repulsion_energy as nuclear_repulsion_energy,
)


class IntegralStrategy(enum.IntEnum):
    DIRECT = 0  # Re-computes integrals every step.
    CACHED = 1  # Pre-computes O(N^4) tensor once.


SolverParams = AndersonParams | LinearMixingParams


@register_pytree_node_class
@dataclasses.dataclass
class Options:
    solver: SolverParams = dataclasses.field(default_factory=LinearMixingParams)
    integral_strategy: IntegralStrategy = IntegralStrategy.CACHED
    perturbation: types.Scalar = 1e-10
    implicit_diff: bool = False

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (self.perturbation,)
        aux_data = (
            self.solver,
            self.integral_strategy,
            self.implicit_diff,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        solver, integral_strategy, implicit_diff = aux_data
        perturbation = children[0]
        return cls(
            solver=solver,
            integral_strategy=integral_strategy,
            perturbation=perturbation,
            implicit_diff=implicit_diff,
        )


@register_pytree_node_class
@dataclasses.dataclass
class Context:
    basis: BatchedBasis

    nuclear_energy: jax.Array

    # Overlap matrix. shape (n_basis, n_basis)
    S: jax.Array

    # Orthogonalizer matrix. shape (n_basis, n_basis)
    X: jax.Array

    # Core Hamiltonian matrix. shape (n_basis, n_basis)
    H_core: jax.Array

    # Two-electron integrals tensor. Only used if
    # Strategy.CACHED is selected.
    # shape (n_basis, n_basis, n_basis, n_basis)
    V: Optional[jax.Array] = None

    def tree_flatten(self):
        children = (
            self.basis,
            self.nuclear_energy,
            self.S,
            self.X,
            self.H_core,
            self.V,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclasses.dataclass
class Result:
    converged: jax.Array
    iterations: jax.Array

    electronic_energy: jax.Array
    nuclear_energy: jax.Array
    total_energy: jax.Array

    # The basis used in the calculation.
    basis: BatchedBasis

    # Fock matrix eigenvalues.
    # shape (n_basis, )
    orbital_energies: jax.Array

    # The molecular orbital coefficients matrix
    # shape (n_basis, n_basis)
    orbitals: jax.Array

    # The closed shell density matrix.
    # shape (n_basis, n_basis)
    density: jax.Array

    def tree_flatten(self):
        children = (
            self.converged,
            self.iterations,
            self.electronic_energy,
            self.nuclear_energy,
            self.total_energy,
            self.basis,
            self.orbital_energies,
            self.orbitals,
            self.density,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def build_context(basis: BatchedBasis, options: Options) -> Context:
    S = overlap_matrix(basis)
    H_core = core_hamiltonian_matrix(basis)
    nuclear_energy = nuclear_repulsion_energy(basis.atoms)

    V = None
    if options.integral_strategy == IntegralStrategy.CACHED:
        V = two_electron_integrals(basis)

    return Context(
        basis=basis,
        nuclear_energy=nuclear_energy,
        S=S,
        X=orthogonalize_basis(S, perturbation=0.0),
        H_core=H_core,
        V=V,
    )


def build_initial_density(context: Context) -> jax.Array:
    n_basis = context.basis.n_basis
    return jnp.zeros((n_basis, n_basis), dtype=jnp.float64)


def _two_electron_matrix(
    P: jax.Array, context: Context, options: Options
) -> jax.Array:
    if options.integral_strategy == IntegralStrategy.DIRECT:
        G = two_electron_matrix(context.basis, P)
    elif options.integral_strategy == IntegralStrategy.CACHED:
        if context.V is None:
            raise ValueError(
                "Two-electron integrals tensor is not cached in context."
            )
        G = two_electron_matrix_from_integrals(context.V, P)
    else:
        raise ValueError(f"Unknown strategy: {options.integral_strategy}")

    return G


def _fock_diagonalize(
    P: jax.Array, context: Context, options: Options
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Performs the core SCF physics: P -> G -> F -> (C, epsilon).
    Returns (F, C, orbital_energies).
    """
    G = _two_electron_matrix(P, context, options)
    F = context.H_core + G
    orbital_energies, C = solve_roothaan(F, context.X, options.perturbation)

    return F, C, orbital_energies


def build_result(
    P: jax.Array, status: SolverStatus, context: Context, options: Options
) -> Result:
    F, C, orbital_energies = _fock_diagonalize(P, context, options)
    electronic_energy_val = electronic_energy(context.H_core, F, P)
    total_energy = electronic_energy_val + context.nuclear_energy
    return Result(
        converged=jnp.asarray(status.converged),
        iterations=jnp.asarray(status.iteration),
        electronic_energy=electronic_energy_val,
        nuclear_energy=context.nuclear_energy,
        total_energy=total_energy,
        basis=context.basis,
        orbital_energies=orbital_energies,
        orbitals=C,
        density=P,
    )


def scf_step(P: jax.Array, context: Context, options: Options) -> jax.Array:
    _, C, _ = _fock_diagonalize(P, context, options)
    return closed_shell_matrix(C, context.basis.n_electrons)


def _build_basis(
    system: BatchedBasis | Molecule,
) -> BatchedBasis:
    if isinstance(system, BatchedBasis):
        return system
    elif isinstance(system, Molecule):
        return BatchedBasis.from_molecule(system)
    else:
        raise TypeError(
            f"Expected input of type BatchedBasis or Molecule, got {type(system)}"
        )


def _solve(
    P0: jax.Array, context: Context, options: Options
) -> tuple[jax.Array, SolverStatus]:
    def step(P: jax.Array) -> jax.Array:
        """The fixed point mapping: P -> P"""
        return scf_step(P, context, options)

    if isinstance(options.solver, AndersonParams):
        return anderson_solve(step, P0, options.solver)
    elif isinstance(options.solver, LinearMixingParams):
        return lm_solve(step, P0, options.solver)
    else:
        raise ValueError(f"Unknown solver params type: {type(options.solver)}")


def _solve_implicit(
    P0: jax.Array, context: Context, options: Options
) -> tuple[jax.Array, SolverStatus]:
    """Implicit differentiation solver."""

    def fixed_point_step(P, ctx):
        """The fixed point mapping: (P, context) -> P"""
        return scf_step(P, ctx, options)

    def primal_solver(ctx):
        """The primal solver. context -> (P, aux)"""
        return _solve(P0, ctx, options)

    solve_fn = attach_implicit_grad(
        fixed_point_step, primal_solver, has_aux=True
    )

    return solve_fn(context)


def solve(
    system: BatchedBasis | Molecule, options: Options = Options()
) -> Result:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        A Result object containing the final energy and orbital coefficients.
    """
    basis = _build_basis(system)
    context = build_context(basis, options)
    P0 = build_initial_density(context)

    if options.implicit_diff:
        P, status = _solve_implicit(P0, context, options)
    else:
        P, status = _solve(P0, context, options)

    return build_result(P, status, context, options)
