from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from slaterform.fixed_point import implicit


class FakeTree(NamedTuple):
    x: float | jax.Array
    y: float | jax.Array


def test_scalar_fixed_point():
    def scalar_step(x, p):
        return jnp.asarray(0.5 * x + p)

    # Fixed point is x = 2 * p
    p_val = 5.0
    expected_x = 10.0
    expected_grad = 2.0

    def fake_solver(p):
        return expected_x

    implicit_solver = implicit.attach_implicit_grad(scalar_step, fake_solver)

    grad_fn = jax.grad(implicit_solver)
    grad = grad_fn(p_val)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-8)


def test_scalar_with_aux_fixed_point():
    def scalar_step(x, p):
        return jnp.asarray(0.5 * x + p)

    # Fixed point is x = 2 * p
    p_val = 5.0
    expected_aux = 6.0
    expected_x = 10.0
    expected_grad = 2.0

    def fake_solver(p):
        return expected_x, expected_aux

    implicit_solver = implicit.attach_implicit_grad(
        scalar_step, fake_solver, has_aux=True
    )

    grad_fn = jax.value_and_grad(implicit_solver, has_aux=True)
    (x, aux), grad = grad_fn(p_val)

    np.testing.assert_allclose(x, expected_x, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(aux, expected_aux, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-8)


def test_pytree_fixed_point():
    # Step: (x, y) -> (0.5 * x + p, 0.25 * y + p)
    # Fixed point is: x = 2 * p, y = (4/3) * p
    def tree_step(state: FakeTree, p):
        return FakeTree(0.5 * state.x + p, 0.25 * state.y + p)

    p_val = 3.0
    expected_tree = FakeTree(6.0, 4.0)
    expeced_tree_grad = FakeTree(2.0, 4.0 / 3.0)

    def fake_solver(p):
        return expected_tree

    implicit_solver = implicit.attach_implicit_grad(tree_step, fake_solver)
    grad_fn = jax.jacrev(implicit_solver)
    grad = grad_fn(p_val)

    np.testing.assert_allclose(
        grad.x, expeced_tree_grad.x, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        grad.y, expeced_tree_grad.y, rtol=1e-5, atol=1e-8
    )


def test_scalar_nonlinear_point():
    # We use the problem: x = x^2 + p
    # For each p there are two fixed points, one of them is:
    # x = (1 + sqrt(1 - 4p)) / 2
    #
    # Differentiating:
    # dx/dp = -1 / sqrt(1 - 4p)

    def scalar_step(x, p):
        return jnp.asarray((x**2) + p)

    # Fixed point is x = 2 * p
    p_val = 0.1
    expected_x = (1.0 + jnp.sqrt(1.0 - 4.0 * p_val)) / 2.0
    expected_grad = -1.0 / jnp.sqrt(1.0 - 4.0 * p_val)

    def fake_solver(p):
        return expected_x

    implicit_solver = implicit.attach_implicit_grad(scalar_step, fake_solver)

    grad_fn = jax.grad(implicit_solver)
    grad = grad_fn(p_val)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-8)


def test_vector_fixed_point():
    # Problem: x = M*x + p  ->  x = (I-M)^-1 * p
    M = jnp.array([[0.5, 0.2], [0.1, 0.4]])
    I = jnp.eye(2)
    inv_factor = jnp.linalg.inv(I - M)  # The analytical Jacobian

    p = jnp.array([0.5, 1.0])
    expected_x = inv_factor @ p
    expected_jacobian = inv_factor

    def vector_step(x, p):
        return M @ x + p

    def fake_solver(p):
        return expected_x

    implicit_solver = implicit.attach_implicit_grad(vector_step, fake_solver)

    grad_fn = jax.jacrev(implicit_solver)
    jacobian = grad_fn(p)
    np.testing.assert_allclose(
        jacobian, expected_jacobian, rtol=1e-5, atol=1e-8
    )
