import dataclasses
from typing import TypeAlias

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
