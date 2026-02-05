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
class AndersonState:
    k: jax.Array  # Current iteration
    x_curr: jax.Array  # Current estimate. Shape: arbitrary
    fx_curr: jax.Array  # f(x_curr). Shape: x_curr.shape
    X_hist: jax.Array  # History of x values. Shape: (m x x0.size)
    F_hist: jax.Array  # History of f(x) values. Shape: (m x x0.size)

    def tree_flatten(self):
        children = (
            self.k,
            self.x_curr,
            self.fx_curr,
            self.X_hist,
            self.F_hist,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple,
    ) -> "AndersonState":
        return cls(*children)


@register_pytree_node_class
@dataclasses.dataclass
class AndersonParams:
    max_iter: int = 50  # Maximum iterations
    m: types.IntScalar = 5  # History size
    lam: types.Scalar = 1e-4  # Regularization parameter
    tol: types.Scalar = 1e-05  # Convergence tolerance
    beta: types.Scalar = 1.0  # Mixing dampener
    static_loop: bool = False  # If True, use a fixed number of iterations
    callback: Optional[Callable[[SolverStatus], None]] = None

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (self.lam, self.tol, self.beta)
        aux_data = (self.max_iter, self.m, self.static_loop, self.callback)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple,
        children: tuple,
    ) -> "AndersonParams":
        max_iter, m, static_loop, callback = aux_data
        lam, tol, beta = children
        return cls(
            max_iter=max_iter,
            m=m,
            lam=lam,
            tol=tol,
            beta=beta,
            static_loop=static_loop,
            callback=callback,
        )


def _step(
    f: Callable[[jax.Array], jax.Array],
    state: AndersonState,
    params: AndersonParams,
) -> AndersonState:
    m = params.m

    # Roll old history back, put x_curr at index 0.
    # I.e: [x_k, x_{k-1}, ..., x_{k-m+1}] -> [x_{k+1}, x_k, ..., x_{k-m+2}]
    # Shape: (m, size)
    X_hist = jnp.roll(state.X_hist, 1, axis=0)
    X_hist = X_hist.at[0].set(state.x_curr.reshape(-1))
    F_hist = jnp.roll(state.F_hist, 1, axis=0)
    F_hist = F_hist.at[0].set(state.fx_curr.reshape(-1))

    # Calculate residuals: G = F - X
    # Shape: (m, size)
    G = F_hist - X_hist

    # Construct Gram matrix (G^T G)
    # Shape: (m, m)
    GTG = G @ G.T

    # If k < m, we only have k+1 valid entries.
    # We want to solve the system only for the top-left (k+1) block.
    # We add ridge regularization to the top left (k+1) block and
    # set the rest to identity.
    hist_len = jnp.minimum(state.k + 1, m)
    mask = jnp.arange(m) < hist_len
    diag_reg = jnp.where(mask, params.lam, 1.0)
    GTG = GTG + jnp.diag(diag_reg)

    # Constraint vector (sum alpha_i = 1)
    # Shape: (m, 1)
    C = jnp.where(mask, 1.0, 0.0)[:, None]

    # Build system matrix H
    # [ GTG   C ]
    # [ C.T   0 ]
    # Shape: (m+1, m+1)
    H = jnp.block([[GTG, C], [C.T, jnp.zeros((1, 1))]])

    # [0, ..., 0, 1]
    # Shape: (m+1,)
    rhs = jnp.append(jnp.zeros(m, dtype=state.x_curr.dtype), 1.0)

    # Solve H * sol = rhs
    sol = jnp.linalg.solve(H, rhs)
    alpha = sol[:m]

    # Compute x_next.
    # Formula: x_next = sum_i(alpha_i * (beta * f(x_i) + (1-beta) * x_i))

    # Shape: (m, size)
    mixed_hist = params.beta * F_hist + (1.0 - params.beta) * X_hist

    # Shape: (size,)
    x_next_flat = alpha @ mixed_hist

    x_next = x_next_flat.reshape(state.x_curr.shape)
    fx_next = f(x_next)

    return AndersonState(state.k + 1, x_next, fx_next, X_hist, F_hist)


def solve(
    f: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    params: AndersonParams,
) -> tuple[jax.Array, SolverStatus]:
    """
    A JIT-compatible Anderson fixed point solver.

    Args:
        f: The fixed point function x_new = f(x).
        x0: Initial guess.
        params: Anderson solver parameters.
    """
    m = params.m
    size = x0.size

    init_state = AndersonState(
        k=jnp.array(0),
        x_curr=x0,
        fx_curr=f(x0),
        X_hist=jnp.zeros((m, size), dtype=x0.dtype),
        F_hist=jnp.zeros((m, size), dtype=x0.dtype),
    )

    step_fn = functools.partial(_step, f)
    final_state, final_status = fixed_point_run(
        step_fn=step_fn, init_state=init_state, params=params
    )

    return final_state.x_curr, final_status
