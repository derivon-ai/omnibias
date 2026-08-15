# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic BPST instanton arrays (regular gauge) for classical Yang-Mills tests.

The connection and its first / second derivatives are closed form in numpy, so
docs snippets and the covariant-jet wrapper can share one gold-standard field
without importing the test tree.
"""

from __future__ import annotations

import numpy as np
from omnibias.geometry.gauge._core import forms


def thooft_eta() -> np.ndarray:
    """Self-dual 't Hooft symbol ``eta_{a mu nu}`` (a=0..2, mu,nu=0..3).

    Self-dual under ``eps_{0123} = +1`` (time at index 0).
    """
    eta = np.zeros((3, 4, 4))
    eps3 = forms.levi_civita_symbol(3)
    for a in range(3):
        for i in range(1, 4):
            for j in range(1, 4):
                eta[a, i, j] = eps3[a, i - 1, j - 1]
        for j in range(1, 4):
            eta[a, 0, j] = 1.0 if a == (j - 1) else 0.0
            eta[a, j, 0] = -1.0 if a == (j - 1) else 0.0
    return eta


def bpst_instanton_arrays(
    points: np.ndarray, rho: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""BPST instanton ``A``, ``dA``, ``ddA`` at ``points`` (regular gauge, g=1).

    ``A_mu^a = 2 eta_{a mu nu} x_nu / (x^2 + rho^2)``.

    Returns ``A[B, mu, a]``, ``dA[B, rho, nu, a] = d_rho A_nu^a`` and
    ``ddA[B, rho, mu, nu, a] = d_rho d_mu A_nu^a``.
    """
    eta = thooft_eta()  # (a, mu, nu)
    x = np.asarray(points, dtype=np.float64)
    d = (x**2).sum(1) + rho**2  # (B,)
    d1 = d[:, None, None]
    d2 = d[:, None, None, None]
    a_arr = 2.0 * np.einsum("anb,Bb->Bna", eta, x) / d1
    s = np.einsum("anb,Bb->Ban", eta, x)
    term1 = 2.0 * np.transpose(eta, (2, 1, 0))[None] / d2
    term2 = -4.0 * np.einsum("Ban,Br->Brna", s, x) / d2**2
    da_arr = term1 + term2
    eye4 = np.eye(4)
    d2b = d[:, None, None, None, None]
    t1 = -4.0 * np.einsum("anr,Bs->Bsrna", eta, x) / d2b**2
    t2 = -4.0 * np.einsum("ans,Br->Bsrna", eta, x) / d2b**2
    t3 = -4.0 * np.einsum("Ban,sr->Bsrna", s, eye4) / d2b**2
    t4 = 16.0 * np.einsum("Ban,Br,Bs->Bsrna", s, x, x) / d2b**3
    dda_arr = t1 + t2 + t3 + t4
    return a_arr, da_arr, dda_arr


__all__ = [
    "bpst_instanton_arrays",
    "thooft_eta",
]
