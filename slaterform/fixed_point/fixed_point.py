from typing import NamedTuple, Callable, Optional, Protocol, TypeVar
import functools
import jax
import jax.numpy as jnp

from slaterform import types


class SolverStatus(NamedTuple):
    iteration: jax.Array
    err: jax.Array
    converged: bool


class SolverResult(NamedTuple):
    x: jax.Array
    status: SolverStatus


class SolverState(Protocol):
    k: jax.Array
    x_curr: jax.Array
    fx_curr: jax.Array


class SolverParams(Protocol):
    max_iter: int
    tol: types.Scalar
    static_loop: bool
    callback: Optional[Callable[[SolverStatus], None]]


S = TypeVar("S", bound=SolverState)
P = TypeVar("P", bound=SolverParams)


def _get_solver_status(
    state: SolverState, params: SolverParams
) -> SolverStatus:
    err = jnp.linalg.norm(state.fx_curr - state.x_curr)
    converged = err < params.tol
    return SolverStatus(iteration=state.k, err=err, converged=converged)


def _should_continue(state: SolverState, params: SolverParams) -> jax.Array:
    """Generic convergence check: ||f(x) - x|| > tol."""
    tol_sq = jnp.square(params.tol)
    err_sq = jnp.sum(jnp.square(state.fx_curr - state.x_curr))
    return jnp.logical_and((err_sq > tol_sq), (state.k < params.max_iter))


def _step_wrapper(state: S, step_fn: Callable[[S, P], S], params: P) -> S:
    new_state = step_fn(state, params)

    if params.callback is not None:
        status = _get_solver_status(new_state, params)
        jax.debug.callback(params.callback, status)

    return new_state


def _scan_body(
    carry: S, _: None, step_fn: Callable[[S, P], S], params: P
) -> tuple[S, None]:
    new_carry = _step_wrapper(carry, step_fn, params)
    return new_carry, None


def run(
    step_fn: Callable[[S, P], S],
    init_state: S,
    params: P,
) -> tuple[S, SolverStatus]:
    """
    Executes the fixed point loop (While or Scan).

    Args:
        step_fn: Function(state, params) -> state
        init_state: The initial state.
        params: The solver parameters.
    """
    if params.static_loop:
        scan_fn = functools.partial(_scan_body, step_fn=step_fn, params=params)
        final_state, _ = jax.lax.scan(
            scan_fn, init_state, length=params.max_iter
        )

    else:
        step_wrapper = functools.partial(
            _step_wrapper, step_fn=step_fn, params=params
        )
        cond_fn = functools.partial(_should_continue, params=params)
        final_state = jax.lax.while_loop(cond_fn, step_wrapper, init_state)

    final_status = _get_solver_status(final_state, params)
    return final_state, final_status
