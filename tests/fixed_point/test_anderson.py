import dataclasses
import functools
from typing import Callable

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from slaterform.fixed_point import anderson
from slaterform.fixed_point import fixed_point


@dataclasses.dataclass
class _TestCase:
    f: Callable[[jax.Array], jax.Array]
    x0: jax.Array
    params: anderson.AndersonParams


def _logger(status: fixed_point.SolverStatus):
    print("Iteration", status.iteration, "Error", status.err)


def newton_step(x, p):
    """The Newton step for solving x^2 - p = 0."""
    return 0.5 * (x + p / x)


def get_linear_test_case(dim=10):
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)

    # Generate a symmetric positive definite matrix A
    M = jax.random.normal(k1, (dim, dim))
    A = M.T @ M + 0.1 * jnp.eye(dim)

    # Generate a random target b
    b = jax.random.normal(k2, (dim,))

    # Define the fixed point function. Choose alpha to ensure stability.
    eigenvalues = jnp.linalg.eigvalsh(A)
    max_eig = jnp.max(eigenvalues)
    alpha = 1.0 / max_eig

    def f(x):
        return x - alpha * (A @ x - b)

    params = anderson.AndersonParams(
        m=dim,
        tol=1e-8,
        lam=1e-14,
        beta=1.0,
        callback=_logger,
    )
    return _TestCase(f=f, x0=jnp.zeros((dim,)), params=params)


@pytest.mark.parametrize(
    "case",
    [
        _TestCase(
            f=lambda x: jnp.asarray(0.5 * x + 1.0),
            x0=jnp.asarray(0.0),
            params=anderson.AndersonParams(
                m=2,
                tol=1e-8,
                lam=0.0,
                beta=1.0,
                callback=_logger,
            ),
        ),
        _TestCase(
            f=lambda x: jnp.cos(x),
            x0=jnp.asarray(0.0),
            params=anderson.AndersonParams(
                m=2,
                tol=1e-8,
                lam=0.0,
                beta=1.0,
                callback=_logger,
            ),
        ),
        _TestCase(
            f=lambda x: newton_step(x, p=2.0),
            x0=jnp.asarray(1.0),
            params=anderson.AndersonParams(
                m=2,
                tol=1e-8,
                lam=0.0,
                beta=1.0,
                callback=_logger,
            ),
        ),
        _TestCase(
            f=lambda x: newton_step(x, p=2.0),
            x0=jnp.asarray(1.0),
            params=anderson.AndersonParams(
                m=2,
                tol=1e-8,
                lam=1e-14,
                beta=1.0,
                static_loop=True,
                callback=_logger,
            ),
        ),
        get_linear_test_case(dim=10),
        get_linear_test_case(dim=20),
    ],
)
def test_solve(case: _TestCase):
    solver_fn = functools.partial(
        anderson.solve,
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
    params = anderson.AndersonParams(
        m=2, tol=1e-8, lam=1e-14, beta=1.0, static_loop=True
    )

    def anderson_sqrt(p):
        x, _ = anderson.solve(
            f=lambda x: newton_step(x, p),
            x0=jnp.asarray(1.0),
            params=params,
        )
        return x

    p_val = 2.0
    expected_val = jnp.sqrt(p_val)
    expected_grad = 1 / (2 * jnp.sqrt(p_val))

    value_and_grad_fn = jax.jit(jax.value_and_grad(anderson_sqrt))
    val, grad = value_and_grad_fn(p_val)
    np.testing.assert_allclose(val, expected_val, atol=1e-6)
    np.testing.assert_allclose(grad, expected_grad, atol=1e-6)
