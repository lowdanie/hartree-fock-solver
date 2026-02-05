import dataclasses
import functools
from typing import Callable

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from slaterform.fixed_point import linear_mixing
from slaterform.fixed_point import fixed_point

from tests.jax_utils import pytree_utils


@dataclasses.dataclass
class _TestCase:
    f: Callable[[jax.Array], jax.Array]
    x0: jax.Array
    params: linear_mixing.LinearMixingParams


def _logger(status: fixed_point.SolverStatus):
    print("Iteration", status.iteration, "Error", status.err)


def newton_step(x, p):
    """The Newton step for solving x^2 - p = 0."""
    return 0.5 * (x + p / x)


def build_linear_step():
    A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    b = jnp.array([1.0, 0.0])
    alpha = 0.2

    def f(x):
        return x - alpha * (A @ x - b)

    return f


def test_linear_mixing_state_pytree():
    state = linear_mixing.LinearMixingState(
        k=jnp.array(5),
        x_curr=jnp.array([1.0, 2.0, 3.0]),
        fx_curr=jnp.array([4.0, 5.0, 6.0]),
    )
    pytree_utils.assert_valid_pytree(state)


def test_linear_mixing_params_pytree():
    params = linear_mixing.LinearMixingParams(
        max_iter=50,
        tol=1e-6,
        damping=0.2,
        static_loop=True,
        callback=_logger,
    )
    pytree_utils.assert_valid_pytree(params)


@pytest.mark.parametrize(
    "case",
    [
        _TestCase(
            f=lambda x: jnp.asarray(0.5 * x + 1.0),
            x0=jnp.asarray(0.0),
            params=linear_mixing.LinearMixingParams(
                tol=1e-8,
                damping=0.1,
                callback=_logger,
            ),
        ),
        _TestCase(
            f=lambda x: jnp.cos(x),
            x0=jnp.asarray(0.0),
            params=linear_mixing.LinearMixingParams(
                tol=1e-8,
                damping=0.1,
                callback=_logger,
            ),
        ),
        _TestCase(
            f=lambda x: newton_step(x, p=2.0),
            x0=jnp.asarray(1.0),
            params=linear_mixing.LinearMixingParams(
                tol=1e-8,
                damping=0.1,
                callback=_logger,
            ),
        ),
        _TestCase(
            f=lambda x: newton_step(x, p=2.0),
            x0=jnp.asarray(1.0),
            params=linear_mixing.LinearMixingParams(
                tol=1e-8,
                damping=0.1,
                static_loop=True,
                callback=_logger,
            ),
        ),
        _TestCase(
            f=build_linear_step(),
            x0=jnp.zeros((2,)),
            params=linear_mixing.LinearMixingParams(
                tol=1e-4,
                damping=0.1,
                static_loop=True,
                callback=_logger,
            ),
        ),
    ],
)
def test_solve(case: _TestCase):
    solver_fn = functools.partial(
        linear_mixing.solve,
        f=case.f,
        params=case.params,
    )
    x, status = jax.jit(solver_fn)(
        x0=case.x0,
    )
    assert status.converged
    err = jnp.linalg.norm(case.f(x) - x)
    assert err < case.params.tol


def test_solve_with_grad():
    params = linear_mixing.LinearMixingParams(
        tol=1e-8,
        damping=0.1,
        static_loop=True,
        callback=_logger,
    )

    def lm_sqrt(p):
        x, _ = linear_mixing.solve(
            f=lambda x: newton_step(x, p),
            x0=jnp.asarray(1.0),
            params=params,
        )
        return x

    p_val = 2.0
    expected_val = jnp.sqrt(p_val)
    expected_grad = 1 / (2 * jnp.sqrt(p_val))

    value_and_grad_fn = jax.jit(jax.value_and_grad(lm_sqrt))
    val, grad = value_and_grad_fn(p_val)
    np.testing.assert_allclose(val, expected_val, atol=1e-6)
    np.testing.assert_allclose(grad, expected_grad, atol=1e-6)
