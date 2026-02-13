import dataclasses
from typing import Any, TypeAlias

import jax
from jax import numpy as jnp
import numpy as np

# Dynamic data
Array: TypeAlias = np.ndarray | jax.Array

# Static metadata
StaticArray: TypeAlias = np.ndarray

IntScalar: TypeAlias = int | jax.Array
Scalar: TypeAlias = float | jax.Array


def promote_dataclass_fields(obj):
    """Converts all Array/StaticArray fields to jax/numpy arrays."""
    valid_data_types = (int, float, complex, list, tuple, np.ndarray, jax.Array)

    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)

        if value is None:
            continue

        if not isinstance(value, valid_data_types):
            continue

        if field.type in (Array, IntScalar, Scalar):
            setattr(obj, field.name, jnp.asarray(value))
        elif field.type == StaticArray:
            setattr(obj, field.name, np.asarray(value))


def safe_cast(x: Any, dtype: jnp.dtype) -> jax.Array:
    """Casts an object to the specified type if possible."""
    if hasattr(x, "astype"):
        return x.astype(dtype)

    return x


def float_cast_pytree(tree: Any, dtype: jnp.dtype) -> Any:
    """Recursively casts all floating point arrays in a pytree to dtype.

    Leaves that are not floating point arrays are returned as is.
    """

    def _cast_leaf(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree.map(_cast_leaf, tree)
