import jax
import jax.numpy as jnp
import numpy as np

import slaterform.types as types


def test_float_cast_pytree():
    # Setup inputs
    f32_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    f64_arr = jnp.array([3.0, 4.0], dtype=jnp.float64)
    int_arr = jnp.array([1, 2], dtype=jnp.int32)
    bool_arr = jnp.array([True, False])

    tree = {
        "floats": [f32_arr, f64_arr],
        "mixed": (int_arr, f32_arr),
        "ignored": [bool_arr, "string", None],
    }

    f16_tree = types.float_cast_pytree(tree, jnp.float16)

    # Check floats were cast
    assert f16_tree["floats"][0].dtype == jnp.float16
    assert f16_tree["floats"][1].dtype == jnp.float16
    assert f16_tree["mixed"][1].dtype == jnp.float16

    # Check non-floats were not caches
    assert f16_tree["mixed"][0].dtype == jnp.int32
    assert f16_tree["ignored"][0].dtype == bool_arr.dtype
    assert f16_tree["ignored"][1] == "string"
    assert f16_tree["ignored"][2] is None

    # Verify values are preserved (approx)
    assert jnp.allclose(f16_tree["floats"][0], f32_arr.astype(jnp.float16))
    assert jnp.allclose(f16_tree["floats"][1], f64_arr.astype(jnp.float16))
    assert jnp.allclose(f16_tree["mixed"][1], f32_arr.astype(jnp.float16))
