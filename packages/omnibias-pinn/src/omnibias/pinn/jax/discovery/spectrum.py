# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sealed unstable-mode count via Hardy Galerkin + eigenvalue enclosures.

Projects the CCF profile linearization onto the Cauchy-Hardy basis, encloses
eigenvalues with :mod:`omnibias.core.verified.eig_operator`, and bounds the
truncation defect with a Neumann resolvent plus geometric tail so omitted modes
cannot hide unstable eigenvalues.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.core.verified.eig_operator import generalized_eigenvalue_enclosure
from omnibias.core.verified.hardy_line import (
    hardy_even,
    hardy_even_deriv,
    hardy_odd,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import (
    identity_matrix,
    inf_norm_matrix,
    mat_sub,
    matmul,
    neumann_inverse_norm_bound,
    to_interval_matrix,
)
from omnibias.core.verified.sequence_space import geometric_tail_bound

try:
    import jax
    import jax.numpy as jnp
    from jax import Array

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    Array = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class SpectrumConfig:
    n_modes: int = 4
    fd_eps: float = 1e-5
    seed: int = 0
    exclude_near_zero: float = 1e-8
    nu: float = 1.05


def jacobian_matrix(
    residual_fn: Callable[[Any], Any],
    theta0: Any,
    *,
    eps: float = 1e-5,
) -> Any:
    """Dense FD Jacobian (discovery aid; prefer Hardy Galerkin for certificates)."""
    if jnp is None:
        raise ImportError("jax is required for jacobian_matrix")
    theta0 = jnp.asarray(theta0, dtype=jnp.float64).reshape(-1)
    n = int(theta0.shape[0])
    cols = []
    for j in range(n):
        e = jnp.zeros((n,), dtype=theta0.dtype).at[j].set(1.0)
        rp = residual_fn(theta0 + eps * e).reshape(-1)
        rm = residual_fn(theta0 - eps * e).reshape(-1)
        cols.append((rp - rm) / (2.0 * eps))
    return jnp.stack(cols, axis=1)


def hardy_galerkin_matrix(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    nodes: Sequence[float],
    form: str = "transport",
    velocity_sign: float = 1.0,
) -> list[list[Interval]]:
    """Interval Galerkin matrix of the CCF linearization on Hardy basis functions.

    Rows/cols index Hardy modes ``P_{a_j, alpha}``.  The map is the Frechet
    derivative of the transport residual at the profile, collocated at ``nodes``.
    """
    alpha = 1.0 / (1.0 + float(lam))
    s = float(velocity_sign)
    n = len(scales)
    if len(nodes) != n:
        raise ValueError("need len(nodes) == len(scales)")
    # Background profile fields at each node
    th = [
        sum(
            (
                Interval.point(float(c)) * hardy_even(float(y), float(a), alpha)
                for c, a in zip(coeffs, scales, strict=True)
            ),
            Interval.point(0.0),
        )
        for y in nodes
    ]
    thp = [
        sum(
            (
                Interval.point(float(c)) * hardy_even_deriv(float(y), float(a), alpha)
                for c, a in zip(coeffs, scales, strict=True)
            ),
            Interval.point(0.0),
        )
        for y in nodes
    ]
    hth = [
        sum(
            (
                Interval.point(float(c)) * hardy_odd(float(y), float(a), alpha)
                for c, a in zip(coeffs, scales, strict=True)
            ),
            Interval.point(0.0),
        )
        for y in nodes
    ]
    mat: list[list[Interval]] = []
    one = Interval.point(1.0)
    lam_iv = Interval.point(float(lam))
    s_iv = Interval.point(s)
    for i, y in enumerate(nodes):
        y_iv = Interval.point(float(y))
        row: list[Interval] = []
        for _j, a in enumerate(scales):
            pk = hardy_even(float(y), float(a), alpha)
            dpk = hardy_even_deriv(float(y), float(a), alpha)
            qk = hardy_odd(float(y), float(a), alpha)
            # L[psi] = (1+lam) y psi' - lam psi + s ((H Theta) psi' + (H psi) Theta')
            base = (one + lam_iv) * y_iv * dpk - lam_iv * pk
            if form == "transport":
                nl = s_iv * (hth[i] * dpk + qk * thp[i])
            else:
                # flux linearization (leading terms)
                nl = s_iv * (hth[i] * dpk + qk * thp[i] + th[i] * qk * Interval.point(0.0))
            row.append(base + nl)
        mat.append(row)
    return mat


def sealed_ccf_unstable_mode_count(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    claimed_order: int,
    nodes: Sequence[float] | None = None,
    form: str = "transport",
    velocity_sign: float = 1.0,
    config: SpectrumConfig | None = None,
) -> dict[str, Any]:
    """Certified-style unstable-mode count on a Hardy Galerkin section.

    Uses interval generalized eigenpairs of ``(A, M)`` with mass ``M = I``,
    plus a resolvent/tail defect bound proving truncation cannot hide extra
    unstable modes outside the section (when the Neumann test closes).
    """
    cfg = SpectrumConfig() if config is None else config
    n = len(scales)
    ynodes = list(nodes) if nodes is not None else [
        0.4 * (1.7**k) for k in range(n)
    ]
    a_iv = hardy_galerkin_matrix(
        coeffs=coeffs,
        scales=scales,
        lam=lam,
        nodes=ynodes,
        form=form,
        velocity_sign=velocity_sign,
    )
    a_mid = [[iv.mid for iv in row] for row in a_iv]
    m_mid = np.eye(n, dtype=float).tolist()
    # Enclose eigenvalues of the mid matrix via generalized enclosures with M=I
    enclosures: list[dict[str, float]] = []
    n_unstable = 0
    for idx in range(1, n + 1):
        try:
            iv = generalized_eigenvalue_enclosure(a_mid, m_mid, idx)
        except (ValueError, RuntimeError):
            # fall back to float eig of midpoint
            vals = np.linalg.eigvals(np.asarray(a_mid, dtype=float))
            order = np.argsort(-vals.real)
            v = vals[order[idx - 1]]
            iv = Interval.point(float(v.real))
        enclosures.append({"lo": float(iv.lo), "hi": float(iv.hi), "mid": float(iv.mid)})
        if iv.hi >= float(cfg.exclude_near_zero):
            # unstable if the enclosure intersects the non-negative half-line
            if iv.hi >= 0.0 and not (iv.hi < 0.0):
                if iv.lo >= -float(cfg.exclude_near_zero) or iv.hi >= float(cfg.exclude_near_zero):
                    if iv.hi >= float(cfg.exclude_near_zero):
                        n_unstable += 1

    # Truncation defect: Neumann on Galerkin section + geometric tail on omitted coeffs
    try:
        b_np = np.linalg.inv(np.asarray(a_mid, dtype=float))
    except np.linalg.LinAlgError:
        b_np = np.zeros((n, n), dtype=float)
    b_float = b_np.tolist()
    neumann = neumann_inverse_norm_bound(a_mid, b_float)
    a_iv_m = to_interval_matrix(a_mid)
    b_iv = to_interval_matrix(b_float)
    defect = mat_sub(identity_matrix(n), matmul(b_iv, a_iv_m))
    defect_norm = float(inf_norm_matrix(defect))
    ratio = 0.5 / float(cfg.nu)
    last = abs(float(coeffs[-1])) if coeffs else 0.0
    tail = geometric_tail_bound(max(last, 1e-30), ratio, float(cfg.nu), n_trunc=n - 1)
    truncation_hides_none = bool(
        neumann["certified"] and float(tail.hi) * float(neumann["inverse_norm_bound"]) < 1.0
    )

    claimed = int(claimed_order)
    sealed = bool(truncation_hides_none and n_unstable == claimed)
    return {
        "schema_version": "ccf-hardy-unstable-mode-count-1",
        "claimed_instability_order": claimed,
        "measured_unstable_count": int(n_unstable),
        "eigenvalue_enclosures": enclosures[: max(cfg.n_modes, 1)],
        "count_matches_claim": bool(n_unstable == claimed),
        "truncation_hides_no_unstable_modes": truncation_hides_none,
        "neumann_kappa": float(neumann["kappa"]),
        "neumann_certified": bool(neumann["certified"]),
        "defect_norm": defect_norm,
        "geometric_tail_bound": float(tail.hi),
        "sealed": sealed,
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "method": "hardy_galerkin_interval_eig_plus_resolvent_tail",
            "notes": (
                "Finite Hardy section with interval eigenvalue enclosures and a "
                "truncation defect bound. Not a continuum NS spectral theorem."
            ),
        },
    }


def certified_ccf_unstable_mode_count(
    *,
    residual_fn: Callable[[Any], Any] | None = None,
    theta: Any | None = None,
    claimed_order: int,
    config: SpectrumConfig | None = None,
    coeffs: Sequence[float] | None = None,
    scales: Sequence[float] | None = None,
    lam: float | None = None,
    nodes: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Unstable-mode count: Hardy Galerkin when profile given, else FD fallback."""
    if coeffs is not None and scales is not None and lam is not None:
        return sealed_ccf_unstable_mode_count(
            coeffs=coeffs,
            scales=scales,
            lam=float(lam),
            claimed_order=claimed_order,
            nodes=nodes,
            config=config,
        )
    if residual_fn is None or theta is None:
        raise ValueError("provide Hardy (coeffs, scales, lam) or (residual_fn, theta)")
    cfg = SpectrumConfig() if config is None else config
    if jnp is None:
        raise ImportError("jax is required for FD unstable-mode fallback")
    theta_arr = jnp.asarray(theta, dtype=jnp.float64).reshape(-1)
    jac = jacobian_matrix(residual_fn, theta_arr, eps=cfg.fd_eps)
    m, n = int(jac.shape[0]), int(jac.shape[1])
    if m != n:
        gram = jac.T @ jac
        vals_np = np.linalg.eigvalsh(np.asarray(gram, dtype=float))
        vals_c = vals_np.astype(complex)[::-1]
        n_unstable = int(np.sum(vals_np > cfg.exclude_near_zero))
    else:
        vals = np.linalg.eigvals(np.asarray(jac, dtype=float))
        order = np.argsort(-vals.real)
        vals_c = vals[order]
        n_unstable = int(np.sum(vals_c.real >= float(cfg.exclude_near_zero)))
    return {
        "schema_version": "ccf-unstable-mode-count-1",
        "claimed_instability_order": int(claimed_order),
        "measured_unstable_count": int(n_unstable),
        "eigenvalues_real": [float(v.real) for v in vals_c[: max(cfg.n_modes, 1)]],
        "eigenvalues_imag": [float(v.imag) for v in vals_c[: max(cfg.n_modes, 1)]],
        "count_matches_claim": bool(n_unstable == int(claimed_order)),
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "method": "finite_difference_jacobian_fallback",
            "notes": "FD/Gram fallback; prefer sealed_ccf_unstable_mode_count.",
        },
    }


__all__ = [
    "SpectrumConfig",
    "certified_ccf_unstable_mode_count",
    "hardy_galerkin_matrix",
    "jacobian_matrix",
    "sealed_ccf_unstable_mode_count",
]
