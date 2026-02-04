from typing import Callable, TypeVar, Any
import functools

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jax.flatten_util import ravel_pytree

Params = TypeVar("Params")
State = TypeVar("State")

# A generic step function.
StepFn = Callable[[State, Params], State]

# A generic fixed point solver function.
# For a given step function step(state, params)->params,
# the solver_fn should satisfy:
# step(solver(params), params) == solver(params)
FixedPointSolverFn = Callable[[Params], State]


def _solve_linear_system(A, b, tol=1e-5):
    """Solves Ax=b using GMRES with flattening where A is a linear operator."""
    b_flat, unravel_fn = ravel_pytree(b)
    # Initial guess = zero with correct pytree structure
    x0_flat = b_flat - b_flat

    def A_flat(v_flat):
        v = unravel_fn(v_flat)
        Av = A(v)
        Av_flat, _ = ravel_pytree(Av)
        return Av_flat

    x_flat, _ = gmres(A_flat, b_flat, x0=x0_flat, tol=tol)
    return unravel_fn(x_flat)


def attach_implicit_grad(
    step_fn: StepFn,
    solver_fn: FixedPointSolverFn,
    has_aux: bool = False,
):
    """
    Wraps a non-differentiable solver with implicit differentiation.

    Formula:
    d_state^T * J_params(solver) = d_state^T * (I - J_state(step))^-1 * J_params(step)

    Where:
       J_params(solver) = Partial derivative of solver_fn w.r.t state
       J_state(step)   = Partial derivative of step_fn w.r.t state
       J_params(step)  = Partial derivative of step_fn w.r.t params

    We solve for d_state^T * J_params(solver) in two steps:

    1. Solve the adjoint system for u:
         (I - J_state(step))^T * u = d_state
    2. Compute gradients w.r.t params:
         d_params = J_params(step)^T * u

    We denote J_params(step) and J_state(step) by J_params and J_state respectively.

    Args:
        solver_fn: Function that computes the fixed point (the primal pass).
                   Args: (params) -> state
        step_fn:   Function defining the fixed point equation: state = step(state, params).
                   MUST be a true fixed point (input == output at convergence).
                   Args: (state, params) -> state
        has_aux:   Whether the solver_fn returns auxiliary data along with the state.
                   If true, the solver is expected to return (state, aux).
                   The aux data is not differentiated through.
    """

    @jax.custom_vjp
    def _solve(params):
        return solver_fn(params)

    def _solve_fwd(params):
        out_primals = solver_fn(params)
        if has_aux:
            state, _ = out_primals
        else:
            state = out_primals
        return out_primals, (state, params)

    def _solve_bwd(residuals, cotangents):
        state, params = residuals

        if has_aux:
            d_state, _ = cotangents  # Ignore gradient w.r.t aux
        else:
            d_state = cotangents

        # 1. Solve the adjoint system for lambda.
        # A(v) = (I - J_state(step))^T * v
        def A(v):
            _, vjp_fun = jax.vjp(lambda s: step_fn(s, params), state)
            jt_v = vjp_fun(v)[0]
            return jax.tree.map(lambda x, y: x - y, v, jt_v)

        u = _solve_linear_system(A, d_state)

        # 2. Compute gradients w.r.t params:
        # d_params = J_params(step)^T * u
        _, vjp_fun_params = jax.vjp(lambda p: step_fn(state, p), params)
        d_params = vjp_fun_params(u)[0]

        return (d_params,)

    _solve.defvjp(_solve_fwd, _solve_bwd)

    return _solve
