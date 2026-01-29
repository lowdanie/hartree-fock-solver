from typing import Callable, TypeVar, Any
import functools

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


Params = TypeVar("Params")
State = TypeVar("State")

# A generic step function.
StepFn = Callable[[State, Params], State]

# A generic fixed point solver function.
# For a given step function step(state, params)->params,
# the solver_fn should satisfy:
# step(solver(params), params) == solver(params)
FixedPointSolverFn = Callable[[Params], State]


def _implicit_lhs(
    v: State, step_fn: StepFn, state: State, params: Params
) -> State:
    """Computes the linear operator A(v) = (I - J_state(step)) * v."""
    # J_state(step) * v
    _, Jv = jax.jvp(lambda s: step_fn(s, params), (state,), (v,))

    # v - Jv
    return jax.tree.map(lambda x, y: x - y, v, Jv)


def _implicit_rhs(
    d_params: Params, step_fn: StepFn, state: State, params: Params
) -> State:
    """Computes u = J_params(step) * d_params."""
    _, u = jax.jvp(lambda p: step_fn(state, p), (params,), (d_params,))
    return u


def attach_implicit_jvp(
    step_fn: StepFn,
    solver_fn: FixedPointSolverFn,
):
    """
    Wraps a non-differentiable solver with implicit differentiation.

    Formula:
    (I - J_state(step)) * d_state = J_params(step) * d_params

    Where:
       d_params       = Incoming tangent w.r.t params
       d_state        = J_params(solve) * d_params  (The total derivative we want)
       J_state(step)  = Partial derivative of step_fn w.r.t state
       J_params(step) = Partial derivative of step_fn w.r.t params

    Args:
        solver_fn: Function that computes the fixed point (the primal pass).
                   Args: (params) -> state
        step_fn:   Function defining the fixed point equation: state = step(state, params).
                   MUST be a true fixed point (input == output at convergence).
                   Args: (state, params) -> state
    """

    @jax.custom_jvp
    def _solve(params):
        return solver_fn(params)

    @_solve.defjvp
    def _solve_jvp(primals, tangents):
        params = primals[0]
        d_params = tangents[0]

        # Compute the fixed point state.
        state_star = _solve(params)

        # LHS Operator: v -> (I - J_state(step)) * v
        matvec = functools.partial(
            _implicit_lhs, step_fn=step_fn, state=state_star, params=params
        )

        # RHS Vector: u = J_params(step) * d_params
        u = _implicit_rhs(d_params, step_fn, state_star, params)

        # Solve: A * d_state = u
        d_state_star, _ = gmres(matvec, u, tol=1e-5)

        return state_star, d_state_star

    return _solve
