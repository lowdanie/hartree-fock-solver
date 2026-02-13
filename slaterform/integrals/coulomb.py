from typing import NamedTuple
import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma
import numpy as np

from slaterform.integrals.gaussian import GaussianBasis3d
from slaterform.integrals.gaussian import overlap_prefactor_3d


# Use a first order Taylor approximation for the Boys function for
# values below this threshold.
# 1e-12 is chosen because the error of the Taylor approx (x/(2n+3))
# becomes smaller than machine epsilon (~1e-16) around this point.
_SMALL_X_THRESHOLD = 1e-12


def boys(n: int, x: jax.Array) -> jax.Array:
    r"""Computes the Boys function F_n(x).

    Formula used:

    $$
    F_n(x) = \frac{\Gamma(n + 1/2) \cdot P(n + 1/2, x)}{2 x^{n + 1/2}}
    $$

    Where P is the regularized lower incomplete gamma function.

    For small x (x < _SMALL_X_THRESHOLD), we use the Taylor expansion

    $$
    F_n(x) \approx \frac{1}{2n + 1} - \frac{x}{2n + 3}
    $$

    Args:
        n: The order of the Boys function (usually non-negative integer).
        x: The argument (must be non-negative).

    Returns:
        The evaluated function.
    """
    # Avoid division by zero
    safe_x = jnp.where(x < _SMALL_X_THRESHOLD, 1.0, x)

    # Compute the exact formula using the safe x
    numerator = gamma(n + 0.5) * gammainc(n + 0.5, safe_x)
    denominator = 2 * (safe_x ** (n + 0.5))
    val_exact = numerator / denominator

    # Compute the small-x Taylor expansion
    val_limit = (1.0 / (2 * n + 1)) - (x / (2 * n + 3))

    # Select the correct value based on the original x
    return jnp.where(x < _SMALL_X_THRESHOLD, val_limit, val_exact)


def _V_base_case(
    size_n: int,
    g1: GaussianBasis3d,
    g2: GaussianBasis3d,
    s: jax.Array,
    C: jax.Array,
    P: jax.Array,
) -> jax.Array:
    """Computes the degree zero case of the n-th order coulomb integral.

    Formula:
    V[i,0,0,0] = K(a,b,A,B) * boys(i, s * ||P-C||^2)

    Args:
      size_n: Compute V[n,0,0,0] for n < size_n.
      g1: The first Gaussian basis. center=A, exponent=a
      g2: The second Gaussian basis. center=B, exponent=b
      s: The scaling factor.
      C: A position. Shape: (3,)
      P: The overlap center: (aA + bB)/p where p = a+b
        Shape: (3,)


    Returns:
      An array V_base of shape (size_n,) that satisfies:
      V_base[i] = V[i,0,0,0]
    """
    K = overlap_prefactor_3d(g1, g2)
    dist_sq = jnp.sum(jnp.square(P - C))
    indices = jnp.arange(size_n)

    return K * boys(indices, s * dist_sq)


class _VerticalTransferParams(NamedTuple):
    s: jax.Array
    p: jax.Array
    PA: jax.Array  # P - A
    PC: jax.Array  # P - C


def _V_vertical_transfer_step(
    params: _VerticalTransferParams,
    carry: tuple[jax.Array, jax.Array],
    i: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Computes the next recurrence step for the vertical transfer.

    Formula:
    V[n,...,i] =    (P - A)V_new[n, ...,i-1]
                -(s/p)(P-C)V_new[n+1...,i-1]
               +((i-1)/(2p)V_new[n, ...,i-2]
         -(((i-1)s)/(2p^2))V_new[n+1...,i-2]

    Args:
      params: The static parameters.
      carry: A tuple (V[...,i-1], V[...,i-2])
        representing the previous two vertical steps. Both arrays
        have the same number of dimensions and shape (size_n,...).
        For j=i-1,i-2, the values V[n,...,j] are assumed to be valid for
        0 <= n < size_n - j

      i: The current vertical index (recurrence depth).

    Returns:
      A tuple (new_carry, output):
        new_carry: (V[n,...,i], V[n,...,i-1]) for the next step.
          They both have the same shape as the input carry arrays.
          For j=i,i-1, the values V[n,...,j] are assumed to be valid for
          0 <= n < size_n - j

        output: V[n,...,i] to be stacked into the result array.
    """
    s, p, PA, PC = params

    # V[n,...,i-1], V[n,...,i-1]
    v_im1, v_im2 = carry

    # V[n+1,...,i-1]
    v_im1_up = jnp.roll(v_im1, shift=-1, axis=0)

    # V[n+1,...,i-2]
    v_im2_up = jnp.roll(v_im2, shift=-1, axis=0)

    term1 = PA * v_im1
    term2 = -(s / p) * PC * v_im1_up
    term3 = ((i - 1) / (2 * p)) * v_im2
    term4 = -(((i - 1) * s) / (2 * p * p)) * v_im2_up

    v_i = term1 + term2 + term3 + term4

    return ((v_i, v_im1), v_i)


def _V_vertical_transfer(
    V: jax.Array,
    size_new: int,
    s: jax.Array,
    p: jax.Array,
    A: jax.Array,
    C: jax.Array,
    P: jax.Array,
) -> jax.Array:
    """Applies vertical transfer to compute another dimension of V with size
    size_new.

    Args:
      V: An array with shape (size_n,...)
      size_new: The size of the new dimension. 1 <= size_new <= size_n.
      s, p: scaling factors
      A, C, P: positions.
    Returns:
      An array V_new with shape V.shape + (,size_new) that satisfies
      V_new[n,...,0] = V[n,...] and the recursive formula defined in
      _V_vertical_transfer_step.
      The values V[n,...,i] are only valid when 0 <= n < size_n - i.
    """
    params = _VerticalTransferParams(s=s, p=p, PA=P - A, PC=P - C)
    step_fn = functools.partial(_V_vertical_transfer_step, params)

    v_i0 = V
    init_carry = (v_i0, jnp.zeros_like(v_i0))
    indices = jnp.arange(1, size_new)
    _, v_rest = jax.lax.scan(step_fn, init_carry, indices, unroll=True)

    return jnp.concatenate(
        (v_i0[..., None], jnp.moveaxis(v_rest, 0, -1)), axis=-1
    )


def _V(
    g1: GaussianBasis3d,
    g2: GaussianBasis3d,
    s: jax.Array,
    C: jax.Array,
) -> jax.Array:
    """The n-th order Coulomb integral.

    The output has shape:
    (g1.max_degree + 1, g1.max_degree + 1, g1.max_degree + 1)
    """
    a, A = jnp.asarray(g1.exponent), jnp.asarray(g1.center)
    b, B = jnp.asarray(g2.exponent), jnp.asarray(g2.center)
    p = a + b
    P = (a * A + b * B) / p

    # If the max_degree is zero, we can directly return the base case.
    if g1.max_degree == 0:
        # V has shape (1,)
        V = _V_base_case(1, g1, g2, s, C, P)
        return V[0, None, None, None]

    size_d = g1.max_degree + 1

    # The first dimension of V needs to have size 3 * size_d to have enough space
    # for 3 vertical transfers.
    # V has shape (3*size_d,)
    V = _V_base_case(3 * size_d, g1, g2, s, C, P)

    # Loop over x,y,z
    for i in range(3):
        # Before the vertical transfer, V has shape: ((3-i)*size_d,) + i * (size_d)
        # After the vertical transfer V has shape: ((3-i)*size_d,) + (i + 1) * (size_d)
        V = _V_vertical_transfer(V, size_d, s, p, A[i], C[i], P[i])

        if i < 2:
            V = V[:-size_d, ...]

    return V[0, ...]


class _HorizontalTransferParams(NamedTuple):
    AB: jax.Array  # A - B


def _horizontal_transfer_step(
    params: _HorizontalTransferParams, carry: jax.Array, j: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Computes the next recurrence step for the horizontal transfer.

    Formula:
    I[...,i,j] = (A - B)I[...,i,j-1] + I[...,i+1,j-1]

    Args:
      params: The static parameters.
      carry: An array I[...,:,j-1] with shape (...,N).
        The values I[...,i,j-1] are valid for 0 <= i < N - (j-1)
      j: The current horizontal index (recurrence depth).

    Returns:
      (new_carry, new_output)
        new_carry: An array I[...,:,j] with the same shape as the input carry.
          The values I[...,i,j] are valid for 0 <= i < N - j
        new_output: Equal to new_carry.
    """
    AB = params.AB
    I_jm1 = carry
    I_jm1_up = jnp.roll(I_jm1, shift=-1, axis=-1)

    I_j = AB * I_jm1 + I_jm1_up

    return (I_j, I_j)


def _horizontal_transfer(
    I: jax.Array, src_dim: int, size_new: int, A: jax.Array, B: jax.Array
) -> jax.Array:
    """Applies a horizontal transfer from src_dim to add a dimension to I of size
    size_new.

    Args:
      I: An input array with shape (...,size_src,...)
      src_dim: The source dimension. 0 <= src_dim < len(I.shape).
      size_new: The size of the new dimension.
        We assume that size_new <= I.shape[src_dim].
      A, B: positions.
    Returns:
      An array I_new with shape (,...,size_src,...,size_new).
      The src_dim is preserved in place, and the new dimension is at axis -1.

      The values I[...,i,...,j] are only valid when 0 <= i < size_new - j
      where i is an index at src_dim.
    """
    if size_new <= 1:
        return I[..., None]

    # Move src_dim to the end
    I_j0 = jnp.moveaxis(I, src_dim, -1)
    params = _HorizontalTransferParams(AB=A - B)
    indices = jnp.arange(1, size_new)
    step_fn = functools.partial(_horizontal_transfer_step, params)

    # I_rest will have shape (size_new-1, ..., src_size)
    _, I_rest = jax.lax.scan(step_fn, I_j0, indices, unroll=True)

    # Move the scan axis to the end and concatenate with the base case.
    I_new = jnp.concatenate(
        (I_j0[..., None], jnp.moveaxis(I_rest, 0, -1)), axis=-1
    )

    # Move src_dim back
    return jnp.moveaxis(I_new, -2, src_dim)


class _ElectronTransferParams(NamedTuple):
    p: float  # a + b
    q: float  # c + d
    alpha: float  # -(1 / q) * (b * (A - B) + d * (C - D))


def _electron_transfer_step(
    params: _ElectronTransferParams,
    carry: tuple[jax.Array, jax.Array],
    j: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Computes the next recurrence step for the electron transfer.

    Formula:
    I[..., i, j] =
      -(1/q)(b(A - B) + d(C - D))I[...,i, j-1]
      +(i/(2q))I[...,i-1,j-1]
      +((j-1)/(2q)I[...,i,j-2]
      -(p/q)I[...,i+1,j-1]

    Args:
      params: The static parameters.
      carry: A tuple (I[...,:,j-1], I[...,:,j-2]). Both arrays have the same
        shape (...,N).
      For k=j-1,j-2, the values I[...,i,k] are assumed to be valid for 0 <= i < N - k.
      j: The current electron index (recurrence depth).

    Returns:
      (new_carry, new_output)
        new_carry: A tuple (I[...,:,j], I[...,:,j-1]). Both arrays have the same
          shape as the input carry.
          For k=j,j-1, the values I[...,i,k] are valid for 0 <= i < N - k.
        new_output: I[...,:,j]
    """
    p, q, alpha = params

    # I[...,i,j-1], I[...,i,j-2]
    I_jm1, I_jm2 = carry

    # I[...,i+1,j-1]
    I_jm1_up = jnp.roll(I_jm1, shift=-1, axis=-1)

    # I[...,i-1,j-1]
    I_jm1_down = jnp.pad(
        I_jm1[..., :-1], ((0, 0),) * (I_jm1.ndim - 1) + ((1, 0),)
    )

    # i_indices[0,...,0,i] = i
    i_indices = jnp.arange(I_jm1.shape[-1]).reshape(
        (1,) * (I_jm1.ndim - 1) + (-1,)
    )

    term1 = alpha * I_jm1
    term2 = i_indices / (2 * q) * I_jm1_down
    term3 = ((j - 1) / (2 * q)) * I_jm2
    term4 = -(p / q) * I_jm1_up

    I_j = term1 + term2 + term3 + term4

    return (I_j, I_jm1), I_j


def _electron_transfer(
    I: jax.Array,
    src_dim: int,
    size_new: int,
    exponents: jax.Array,
    centers: jax.Array,
) -> jax.Array:
    """Apply an electron transfer from the first electron at src_dim to add a
    new dimension to I of size size_new representing the third electron.

    Args:
      I: An input array with shape (...,size_src,...)
      src_dim: The source dimension. 0 <= src < len(I.shape).
      size_new: The size of the new dimension.
        We assume that size_new <= I.shape[src_dim].
      exponents: The exponents a, b, c, d
      centers: The centers A, B, C, D

    Returns:
      An array I_new with shape (...,size_src,...,size_new).
      The src_dim is preserved in place, and the new dimension is at axis -1.
      The values I[...,i,...,j] are only valid when 0 <= i < size_new - j
      where i is an index at src_dim.
    """
    if size_new <= 1:
        return I[..., None]

    a, b, c, d = exponents
    A, B, C, D = centers

    # Move src_dim to the end
    I_j0 = jnp.moveaxis(I, src_dim, -1)

    q = c + d
    params = _ElectronTransferParams(
        p=a + b,
        q=q,
        alpha=-(1 / q) * (b * (A - B) + d * (C - D)),
    )
    indices = jnp.arange(1, size_new)
    init_carry = (I_j0, jnp.zeros_like(I_j0))
    step_fn = functools.partial(_electron_transfer_step, params)

    # I_rest will have shape (size_new-1, ..., src_size)
    _, I_rest = jax.lax.scan(step_fn, init_carry, indices, unroll=True)

    # Move the scan axis to the end and concatenate with the base case.
    I_new = jnp.concatenate(
        (I_j0[..., None], jnp.moveaxis(I_rest, 0, -1)), axis=-1
    )

    # Move src_dim back
    return jnp.moveaxis(I_new, -2, src_dim)


def _two_electron_base_case(
    g1: GaussianBasis3d,
    g2: GaussianBasis3d,
    g3: GaussianBasis3d,
    g4: GaussianBasis3d,
) -> jax.Array:
    a = jnp.asarray(g1.exponent)
    b = jnp.asarray(g2.exponent)
    c, C = jnp.asarray(g3.exponent), jnp.asarray(g3.center)
    d, D = jnp.asarray(g4.exponent), jnp.asarray(g4.center)

    p = a + b
    q = c + d
    s = (p * q) / (p + q)

    Q = (c * C + d * D) / q

    K = overlap_prefactor_3d(g3, g4)
    alpha = 2 * jnp.power(jnp.pi, 5 / 2) / (p * q * jnp.sqrt(p + q))

    return alpha * K * _V(g1, g2, s, Q)


def one_electron(
    g1: GaussianBasis3d,
    g2: GaussianBasis3d,
    C: jax.Array,
) -> jax.Array:
    """Computes the one electron Coulomb integral with center C.

    Formula:

    G1(x,y,z) = (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
    G2(x,y,z) = (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))

    I[ix,iy,iz,jx,jy,jz] =
        integral
            G1(x,y,z) * G2(x,y,z) / sqrt((x-Cx)^2+(y-Cy)^2+(z-Cz)^2))
        dx dy dz

    Args:
      g1: The first Gaussian basis shell (center A, exponent a, degree d1).
      g2: The second Gaussian basis shell (center B, exponent b, degree d2).
      C: The position of the nuclear center. Shape: (3,)

    Returns:
      An array I of shape (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1).
      The first 3 dimensions correspond to the angular momentum of g1 (x,y,z).
      The last 3 dimensions correspond to the angular momentum of g2 (x,y,z).
    """
    a, A, d1 = jnp.asarray(g1.exponent), jnp.asarray(g1.center), g1.max_degree
    b, B, d2 = jnp.asarray(g2.exponent), jnp.asarray(g2.center), g2.max_degree

    # Pad g1 so that we have enough space to do horizontal transfers.
    padded_g1 = GaussianBasis3d(max_degree=d1 + d2, exponent=a, center=A)

    # Compute the base case I[...,0,0,0]
    I = (2 * jnp.pi / (a + b)) * _V(padded_g1, g2, a + b, C)

    # For x,y,z on the second electron.
    for i in range(3):
        I = _horizontal_transfer(I, src_dim=i, size_new=d2 + 1, A=A[i], B=B[i])

        # Remove the padding that is no longer needed.
        I = I[(slice(0, d1 + 1),) * (i + 1) + (Ellipsis,)]

    return I


def two_electron(
    g1: GaussianBasis3d,
    g2: GaussianBasis3d,
    g3: GaussianBasis3d,
    g4: GaussianBasis3d,
) -> jax.Array:
    """Computes the two electron Coulomb repulsion integral.

    Formula:

    G1(x,y,z) = (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
    G2(x,y,z) = (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))
    G3(x,y,z) = (x-Cx)^kx (y-Cy)^ky (z-Cz)^kz e^(-c((x-Cx)^2+(y-Cy)^2+(z-Cz)^2))
    G4(x,y,z) = (x-Dx)^lx (y-Dy)^ly (z-Dz)^lz e^(-d((x-Dx)^2+(y-Dy)^2+(z-Dz)^2))

    I[ix,iy,iz,jx,jy,jz,kx,ky,kz,lx,ly,lz] =
        integral
            G1(x1,y1,z1) * G2(x1,y1,z1) *
            G3(x2,y2,z2) * G4(x2,y2,z2) /
            sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2)
        dx1 dy1 dz1 dx2 dy2 dz2

    Args:
      g1: The first Gaussian basis shell (center A, exponent a, degree d1).
      g2: The second Gaussian basis shell (center B, exponent b, degree d2).
      g3: The third Gaussian basis shell (center C, exponent c, degree d3).
      g4: The fourth Gaussian basis shell (center D, exponent d, degree d4

    Returns:
        An array I of shape:
        (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1,
         d3+1, d3+1, d3+1, d4+1, d4+1, d4+1)
    """
    # Pad g3 so that we have enough space to do horizontal transfers
    # to g4.
    padded_g3 = GaussianBasis3d(
        max_degree=g3.max_degree + g4.max_degree,
        exponent=g3.exponent,
        center=g3.center,
    )

    # Pad g1 so that we have enough space to do horizontal transfers
    # to g2 and electron transfers to padded_g3.
    padded_g1 = GaussianBasis3d(
        max_degree=g1.max_degree + g2.max_degree + padded_g3.max_degree,
        exponent=g1.exponent,
        center=g1.center,
    )

    # I has shape (d1+d2+d3+d4,) * 3
    I = _two_electron_base_case(padded_g1, g2, padded_g3, g4)

    A = jnp.asarray(g1.center)
    B = jnp.asarray(g2.center)
    C = jnp.asarray(g3.center)
    D = jnp.asarray(g4.center)
    exponents = jnp.array([g1.exponent, g2.exponent, g3.exponent, g4.exponent])

    # Apply an electron transfer from g1 to g3 in x,y,z
    for i in range(3):
        I = _electron_transfer(
            I,
            src_dim=i,
            size_new=padded_g3.max_degree + 1,
            exponents=exponents,
            centers=jnp.array([A[i], B[i], C[i], D[i]]),
        )
        # Remove the padding that is no longer needed.
        I = I[
            (slice(0, g1.max_degree + g2.max_degree + 1),) * (i + 1)
            + (Ellipsis,)
        ]

    # Apply a horizontal transfer from g1 to g2 in x,y,z
    for i in range(3):
        I = _horizontal_transfer(
            I, src_dim=i, size_new=g2.max_degree + 1, A=A[i], B=B[i]
        )

        # Remove the padding that is no longer needed
        I = I[(slice(0, g1.max_degree + 1),) * (i + 1) + (Ellipsis,)]

    # Apply a horizontal transfer from g3 to g4 in x,y,z
    for i in range(3):
        I = _horizontal_transfer(
            I, src_dim=i + 3, size_new=g4.max_degree + 1, A=C[i], B=D[i]
        )

        # Remove the padding that is no longer needed
        I = I[
            (slice(0, g1.max_degree + 1),) * 3
            + (slice(0, g3.max_degree + 1),) * (i + 1)
            + (Ellipsis,)
        ]

    # Reorder the axes from g1, g3, g2, g4 to g1, g2, g3, g4
    I = jnp.moveaxis(I, [3, 4, 5], [6, 7, 8])

    return I
