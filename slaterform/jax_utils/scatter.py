from collections.abc import Sequence
from typing import Optional

import jax
from jax import numpy as jnp


import jax
import jax.numpy as jnp


def _make_window_index(
    start: jax.Array,
    size: int,
    axis: int,
    total_dims: int,
) -> jax.Array:
    """Creates a broadcast-ready index window for a specific dimension.

    Generates indices [start, start+1, ..., start+size-1] reshaped
    to be orthogonal to other dimensions, but aligned with the batch.

    Args:
        start: Start indices. Shape (n_tiles,).
        size: Size of the window in this dimension.
        axis: Which dimension (0 to total_dims-1) this window targets.
        total_dims: Total number of dimensions in the target tensor.

    Returns:
        Index array of shape (n_tiles, 1, ..., size, ..., 1).
    """
    batch_size = start.shape[0]

    # Create flat window: [s, s+1, ..., s+size-1] -> Shape (n_tiles, size)
    offset = jnp.arange(size)
    idx = start[:, None] + offset[None, :]

    # Reshape for broadcasting
    # We put 'size' at (axis + 1) because axis 0 is the batch.
    shape = [batch_size] + [1] * total_dims
    shape[axis + 1] = size

    return idx.reshape(shape)


def add_tiles(
    target: jax.Array,
    tiles: jax.Array,
    starts: Sequence[jax.Array],
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Adds a batch of tiles into a target array at specified offsets.

    Args:
        target: The array to update. Shape (d1,...,dn).
        tiles: The tiles to add. Shape (n_tiles, t1,...,tn).
        starts: Start indices for each dimension. Length n. Each has shape (n_tiles,).
        mask: Tiles where mask==0 are not added.
            Shape (batch,).

    Returns:
        The updated matrix.
    """
    n_dim = target.ndim
    n_tiles = tiles.shape[0]

    if len(starts) != n_dim:
        raise ValueError(
            f"Expected {n_dim} start arrays for a {n_dim}-D target, "
            f"got {len(starts)}."
        )

    # Apply the mask. Zero out values for padded batches.
    if mask is not None:
        mask_shape = (n_tiles,) + (1,) * n_dim
        tiles = tiles * mask.reshape(mask_shape)

    # Create a grid of target tile indices: (n_tiles, d1, d2, ...)
    indices = [
        _make_window_index(start, tiles.shape[i + 1], i, n_dim)
        for i, start in enumerate(starts)
    ]
    return target.at[tuple(indices)].add(tiles.astype(target.dtype))
