from typing import NamedTuple
import functools
import dataclasses

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing import Callable, Optional

from slaterform import types
from slaterform.fixed_point.fixed_point import SolverStatus
from slaterform.fixed_point.fixed_point import run as fixed_point_run


@register_pytree_node_class
@dataclasses.dataclass
class LinearMixingState:
    k: jax.Array  # Current iteration
    x_curr: jax.Array  # Current estimate. Shape: arbitrary
    fx_curr: jax.Array  # f(x_curr). Shape: x_curr.shape

    def tree_flatten(self):
        children = (
            self.k,
            self.x_curr,
            self.fx_curr,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple,
    ) -> "LinearMixingState":
        return cls(*children)


@register_pytree_node_class
@dataclasses.dataclass
class LinearMixingParams:
    max_iter: int = 50  # Maximum iterations
    tol: types.Scalar = 1e-05  # Convergence tolerance
    damping: types.Scalar = 0.0  # Mixing dampener
    static_loop: bool = False  # If True, use a fixed number of iterations
    callback: Optional[Callable[[SolverStatus], None]] = None

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (
            self.tol,
            self.damping,
        )
        aux_data = (self.max_iter, self.static_loop, self.callback)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple,
        children: tuple,
    ) -> "LinearMixingParams":
        max_iter, static_loop, callback = aux_data
        tol, damping = children
        return cls(
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            static_loop=static_loop,
            callback=callback,
        )


def _step(
    f: Callable[[jax.Array], jax.Array],
    state: LinearMixingState,
    params: LinearMixingParams,
) -> LinearMixingState:
    x_curr = state.x_curr
    fx_curr = state.fx_curr
    alpha = params.damping

    x_next = (1 - alpha) * fx_curr + alpha * x_curr
    fx_next = f(x_next)

    return LinearMixingState(
        k=state.k + 1,
        x_curr=x_next,
        fx_curr=fx_next,
    )


def solve(
    f: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    params: LinearMixingParams,
) -> jax.Array:
    """
    Solves the fixed point equation x = f(x) using linear mixing.

    Args:
        f: The fixed point function.
        x0: Initial guess for the fixed point.
        params: LinearMixingParams containing solver parameters.

    Returns:
        The estimated fixed point.
    """
    x_init = f(x0)
    fx_init = f(x_init)
    init_state = LinearMixingState(
        k=jnp.asarray(1),
        x_curr=x_init,
        fx_curr=fx_init,
    )

    step_fn = functools.partial(_step, f)
    final_state = fixed_point_run(
        step_fn=step_fn, init_state=init_state, params=params
    )

    return final_state.x_curr
