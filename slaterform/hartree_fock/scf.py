import dataclasses
from typing import Callable, Optional
import functools
import enum

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


import slaterform.types as types
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
from slaterform.structure.batched_basis import BatchedBasis
from slaterform.structure.molecule import Molecule
from slaterform.structure.nuclear import (
    repulsion_energy as nuclear_repulsion_energy,
)

SolverCallback = Callable[["State"], None]


class IntegralStrategy(enum.IntEnum):
    DIRECT = 0  # Re-computes integrals every step.
    CACHED = 1  # Pre-computes O(N^4) tensor once.


class ExecutionMode(enum.IntEnum):
    # Runs until convergence. Not differentiable.
    CONVERGENCE = 0

    # Runs for a fixed number of iterations. Differentiable.
    FIXED = 1

    # Runs until convergence and is forward differentiable
    # via implicit differentiation.
    IMPLICIT = 2


@register_pytree_node_class
@dataclasses.dataclass
class CallbackOptions:
    interval: types.IntScalar = 10
    func: Optional[SolverCallback] = None

    def __post_init__(self):
        if isinstance(self.interval, int):
            if self.interval < 1:
                raise ValueError("callback_interval must be >= 1")
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (self.interval,)
        aux_data = (self.func,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            interval=children[0],
            func=aux_data[0],
        )


@register_pytree_node_class
@dataclasses.dataclass
class Options:
    max_iterations: int = 50  # Acts as 'n_steps' in FIXED mode
    execution_mode: ExecutionMode = ExecutionMode.CONVERGENCE
    integral_strategy: IntegralStrategy = IntegralStrategy.CACHED

    convergence_threshold: types.Scalar = 1e-6
    perturbation: types.Scalar = 0.0

    # Damping factor [0.0, 1.0).
    # 0.0 means no damping (use 100% new density).
    damping: types.Scalar = 0.0

    callback: CallbackOptions = dataclasses.field(
        default_factory=CallbackOptions
    )

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (
            self.convergence_threshold,
            self.perturbation,
            self.damping,
            self.callback,
        )
        aux_data = (
            self.max_iterations,
            self.execution_mode,
            self.integral_strategy,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            max_iterations=aux_data[0],
            execution_mode=aux_data[1],
            integral_strategy=aux_data[2],
            convergence_threshold=children[0],
            perturbation=children[1],
            damping=children[2],
            callback=children[3],
        )

    @classmethod
    def differentiable(
        cls, steps=25, callback: CallbackOptions = CallbackOptions()
    ) -> "Options":
        """Helper for differentiable optimization."""
        return cls(
            max_iterations=steps,
            execution_mode=ExecutionMode.FIXED,
            integral_strategy=IntegralStrategy.CACHED,
            perturbation=1e-10,  # Safety for gradients
            callback=callback,
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
class State:
    iteration: jax.Array

    # Molecular orbital coefficients matrix. shape (n_basis, n_basis)
    C: jax.Array

    # Closed shell density matrix. shape (n_basis, n_basis)
    P: jax.Array

    # Fock matrix. shape (n_basis, n_basis)
    F: jax.Array

    electronic_energy: jax.Array
    total_energy: jax.Array

    # Fock matrix eigenvalues. shape (n_basis, )
    orbital_energies: jax.Array

    # Change in density matrix. ||P_new - P_old||^2
    delta_P_sq: jax.Array

    def tree_flatten(self):
        children = (
            self.iteration,
            self.C,
            self.P,
            self.F,
            self.electronic_energy,
            self.total_energy,
            self.orbital_energies,
            self.delta_P_sq,
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


def build_initial_state(context: Context, options: Options) -> State:
    n_basis = context.basis.n_basis

    conv_thresh_sq = jnp.square(options.convergence_threshold)

    return State(
        iteration=jnp.array(0, dtype=jnp.int32),
        C=jnp.zeros((n_basis, n_basis), dtype=jnp.float64),
        P=jnp.zeros((n_basis, n_basis), dtype=jnp.float64),
        F=context.H_core,
        electronic_energy=jnp.asarray(0.0, dtype=jnp.float64),
        total_energy=context.nuclear_energy,
        orbital_energies=jnp.zeros(n_basis, dtype=jnp.float64),
        delta_P_sq=jnp.array(conv_thresh_sq + 1.0, dtype=jnp.float64),
    )


def _is_converged(state: State, options: Options) -> jax.Array:
    return state.delta_P_sq <= jnp.square(options.convergence_threshold)


def build_result(state: State, context: Context, options: Options) -> Result:
    return Result(
        converged=_is_converged(state, options),
        iterations=state.iteration,
        electronic_energy=state.electronic_energy,
        nuclear_energy=context.nuclear_energy,
        total_energy=state.total_energy,
        basis=context.basis,
        orbital_energies=state.orbital_energies,
        orbitals=state.C,
        density=state.P,
    )


def _two_electron_matrix(
    state: State, P: jax.Array, context: Context, options: Options
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


def scf_step(state: State, context: Context, options: Options) -> State:
    # Solve for new orbital coefficients and density.
    # C has shape (n_basis, n_basis)
    orbital_energies, C = solve_roothaan(
        state.F, context.X, options.perturbation
    )
    # shape (n_basis, n_basis)
    P_new = closed_shell_matrix(C, context.basis.n_electrons)

    # Damping.
    alpha = jax.lax.select(state.iteration > 0, options.damping, 0.0)
    P = (1.0 - alpha) * P_new + alpha * state.P

    # Compute the Fock matrix and energy for the new density P.
    # shape (n_basis, n_basis)
    G = _two_electron_matrix(state, P, context, options)
    F = context.H_core + G  # shape (n_basis, n_basis)
    electronic_energy_val = electronic_energy(context.H_core, F, P)

    return State(
        iteration=state.iteration + 1,
        C=C,
        P=P,
        F=F,
        electronic_energy=electronic_energy_val,
        total_energy=electronic_energy_val + context.nuclear_energy,
        orbital_energies=orbital_energies,
        delta_P_sq=jnp.sum(jnp.square(P - state.P)),
    )


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


def _should_continue(state: State, options: Options) -> jax.Array:
    return jnp.logical_and(
        jnp.logical_not(_is_converged(state, options)),
        state.iteration < options.max_iterations,
    )


def _maybe_run_callback(state: State, options: CallbackOptions) -> None:
    if options.func is None:
        return

    should_run = state.iteration % options.interval == 0

    def run_callback():
        jax.debug.callback(options.func, state)

    def noop_callback():
        return None

    jax.lax.cond(should_run, run_callback, noop_callback)


@jax.checkpoint
def _perform_step(state: State, context: Context, options: Options) -> State:
    state = scf_step(state, context, options)
    _maybe_run_callback(state, options.callback)

    return state


def _solve_convergence(context: Context, options: Options) -> Result:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        A Result object containing the final energy and orbital coefficients.
    """
    state = build_initial_state(context, options)

    cond_fn = functools.partial(_should_continue, options=options)
    step_fn = functools.partial(_perform_step, context=context, options=options)
    state = jax.lax.while_loop(cond_fn, step_fn, state)

    return build_result(state, context, options)


def _solve_fixed(context: Context, options: Options) -> Result:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        A Result object containing the final energy and orbital coefficients.
    """
    state = build_initial_state(context, options)

    def scan_fn(state, _):
        state = _perform_step(state, context, options)
        return state, None

    state, _ = jax.lax.scan(scan_fn, state, None, length=options.max_iterations)

    return build_result(state, context, options)


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

    if options.execution_mode == ExecutionMode.CONVERGENCE:
        return _solve_convergence(context, options)
    elif options.execution_mode == ExecutionMode.FIXED:
        return _solve_fixed(context, options)
    else:
        raise ValueError(f"Unknown execution mode: {options.execution_mode}")
