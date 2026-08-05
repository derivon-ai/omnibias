# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Boolean spectrum engine (torch) via multivariate jets.

The whole Boolean spectrum is read off the *mixed partial derivatives* of the
multilinear extension. For cube values ``v`` the extension ``F`` is built as a
multivariate jet at a base point with :func:`omnibias.torch.jet_mv.identity_jet`
and the truncated Cauchy product
(:func:`omnibias.core.multi_index.multiply_table`), and then
:func:`omnibias.torch.jet_mv.jet_partials` returns every coefficient at once:

* the ``{0,1}`` **Mobius** coefficient ``m_S = d^{|S|} F / prod_{i in S} d x_i``
  at ``x = 0`` (its reduction mod 2 is the GF(2) ANF), and
* the ``{+-1}`` **Walsh/Fourier** coefficient ``hat f(S)`` from the same machinery
  in the ``chi = 1 - 2x`` basis.

Everything is differentiable in ``v`` (the transform is linear), so these feed the
design losses. The exact :mod:`omnibias.boolean._core` transforms are the ground
truth the tests check against.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.core.multi_index import multiply_table
from omnibias.torch.jet_mv import identity_jet, jet_partials
from torch import Tensor


def _num_vars(size: int) -> int:
    if size < 1 or (size & (size - 1)) != 0:
        raise ValueError(f"values length must be a power of two, got {size}")
    return size.bit_length() - 1


def _as_tensor(values: Sequence[float] | Tensor) -> Tensor:
    if isinstance(values, Tensor):
        return values
    return torch.as_tensor(values, dtype=torch.get_default_dtype())


def _mul_jet(
    p: Tensor, q: Tensor, g_idx: Tensor, a_idx: Tensor, b_idx: Tensor, m: int
) -> Tensor:
    """Truncated multivariate polynomial product via gather + segment scatter-add."""
    contrib = p[a_idx] * q[b_idx]
    out = torch.zeros(m, dtype=p.dtype, device=p.device)
    return out.index_add(0, g_idx, contrib)


def _extension_jet(values: Tensor, mode: str, order: int) -> tuple[Tensor, int]:
    n = _num_vars(values.shape[0])
    x0 = torch.zeros(n, dtype=values.dtype, device=values.device)
    coord = identity_jet(x0, order)
    m = coord.shape[0]
    table = multiply_table(n, order)
    g_idx = torch.tensor([t[0] for t in table], dtype=torch.long, device=values.device)
    a_idx = torch.tensor([t[1] for t in table], dtype=torch.long, device=values.device)
    b_idx = torch.tensor([t[2] for t in table], dtype=torch.long, device=values.device)
    one = torch.zeros(m, dtype=values.dtype, device=values.device)
    one[0] = 1.0
    g1: list[Tensor] = []
    g0: list[Tensor] = []
    for i in range(n):
        c = coord[:, i]
        if mode == "mobius":
            g1.append(c)
            g0.append(one - c)
        elif mode == "walsh":
            g1.append((one - c) * 0.5)
            g0.append((one + c) * 0.5)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    total: Tensor | None = None
    for a in range(1 << n):
        prod = one
        for i in range(n):
            factor = g1[i] if (a >> i) & 1 else g0[i]
            prod = _mul_jet(prod, factor, g_idx, a_idx, b_idx, m)
        term = values[a] * prod
        total = term if total is None else total + term
    assert total is not None
    return total, n


def _coeffs_vector(values: Tensor, mode: str) -> Tensor:
    n = _num_vars(values.shape[0])
    jet, _ = _extension_jet(values, mode, order=n)
    partials = jet_partials(jet, n, n)
    out: list[Tensor | None] = [None] * (1 << n)
    for alpha, val in partials.items():
        if all(e <= 1 for e in alpha):
            mask = 0
            for i, e in enumerate(alpha):
                if e:
                    mask |= 1 << i
            out[mask] = val
    zero = values[0] * 0.0
    return torch.stack([zero if x is None else x for x in out])


def mobius_coeffs(values: Sequence[float] | Tensor) -> Tensor:
    """``{0,1}`` Mobius / multilinear coefficients ``m_S`` (indexed by subset mask)."""
    return _coeffs_vector(_as_tensor(values), "mobius")


def walsh_coeffs(values: Sequence[float] | Tensor) -> Tensor:
    """``{+-1}`` Walsh / Fourier coefficients ``hat f(S)`` (indexed by subset mask).

    Pass the ``{+-1}`` *output* values (e.g. ``1 - 2*tt``) to match the usual
    Fourier normalization.
    """
    return _coeffs_vector(_as_tensor(values), "walsh")


def _to_spectrum(coeffs: Tensor) -> dict[frozenset[int], Tensor]:
    n = _num_vars(coeffs.shape[0])
    return {
        frozenset(i for i in range(n) if (mask >> i) & 1): coeffs[mask]
        for mask in range(1 << n)
    }


def mobius_spectrum(values: Sequence[float] | Tensor) -> dict[frozenset[int], Tensor]:
    """Mobius coefficients as a ``{variable-set: m_S}`` mapping."""
    return _to_spectrum(mobius_coeffs(values))


def walsh_spectrum(values: Sequence[float] | Tensor) -> dict[frozenset[int], Tensor]:
    """Walsh/Fourier coefficients as a ``{variable-set: hat f(S)}`` mapping."""
    return _to_spectrum(walsh_coeffs(values))


def influences_diff(values: Sequence[float] | Tensor) -> Tensor:
    """Per-coordinate influence ``Inf_i = sum_{S in i} hat f(S)^2`` (differentiable).

    Pass ``{+-1}`` output values; equals the exact ``_core`` influences for a
    Boolean function.
    """
    w = walsh_coeffs(values)
    n = _num_vars(w.shape[0])
    out = []
    for i in range(n):
        idx = [mask for mask in range(1 << n) if (mask >> i) & 1]
        out.append((w[idx] ** 2).sum())
    return torch.stack(out)


def algebraic_degree_soft(values: Sequence[float] | Tensor) -> Tensor:
    """Spectral-energy-weighted mean monomial order (a smooth degree proxy).

    ``sum_S |S| hat f(S)^2 / sum_S hat f(S)^2``. This is a differentiable surrogate
    for the integer algebraic degree, not the exact value; the exact degree is
    :func:`omnibias.boolean._core.algebraic_degree`.
    """
    w = walsh_coeffs(values)
    n = _num_vars(w.shape[0])
    orders = torch.tensor(
        [bin(mask).count("1") for mask in range(1 << n)],
        dtype=w.dtype,
        device=w.device,
    )
    energy = (w**2).sum()
    return (orders * w**2).sum() / energy


__all__ = [
    "algebraic_degree_soft",
    "influences_diff",
    "mobius_coeffs",
    "mobius_spectrum",
    "walsh_coeffs",
    "walsh_spectrum",
]
