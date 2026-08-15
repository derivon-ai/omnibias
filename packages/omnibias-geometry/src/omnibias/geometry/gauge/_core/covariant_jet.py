# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge-covariant jet of a connection: ``F`` and ``D F`` only.

A coordinate jet of ``A_mu^a`` stores ``partial_rho A_nu^a``, which transforms
inhomogeneously. This module drops ``A`` / ``dA`` / ``ddA`` after construction
and exposes only the Utiyama / Olver fibers

    F_{mu nu}^a ,   (D_rho F_{mu nu})^a

plus their Weyl singlets. Symbolic regression must search those singlets (or
same-type adjoint residuals), never raw ``partial^alpha A``.

Honesty: classical *local* identities on a smooth connection on flat ``R^d``.
Not a Wilson / Polyakov language, not a continuum mass-gap claim, not quantum
Yang-Mills. The operator columns are closed-form given ``A, dA, ddA``; any
sparse fit on top is a numerical STLSQ step.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.connection import EUCLIDEAN_4D
from omnibias.geometry.gauge._core.forms import levi_civita_symbol
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra

SINGLET_TR_F2 = "tr(F^2)"
SINGLET_TR_F_FTILDE = "tr(F*Ftilde)"
SINGLET_YM_SQ = "|D*F|^2"
SINGLET_BIANCHI_SQ = "|Bianchi|^2"
SINGLET_SELF_DUAL_SQ = "|F-*F|^2"

LEGAL_SINGLET_ATOMS: frozenset[str] = frozenset(
    {
        SINGLET_TR_F2,
        SINGLET_TR_F_FTILDE,
        SINGLET_YM_SQ,
        SINGLET_BIANCHI_SQ,
        SINGLET_SELF_DUAL_SQ,
    }
)

ADJOINT_YM = "D*F"
ADJOINT_BIANCHI = "Bianchi"
LEGAL_ADJOINT_1FORM_ATOMS: frozenset[str] = frozenset({ADJOINT_YM, ADJOINT_BIANCHI})

# On a Euclidean self-dual field, action_density / topological_charge_density
# equals 8 pi^2 (kernel normalizations 1/4 and 1/32 pi^2).
SELF_DUAL_ACTION_OVER_TOPOLOGICAL: float = 8.0 * math.pi**2


def assert_library_gauge_legal(
    names: Iterable[str],
    *,
    allow: frozenset[str] = LEGAL_SINGLET_ATOMS,
) -> None:
    """Raise ``ValueError`` if any name is outside the closed allowlist."""
    illegal = [name for name in names if name not in allow]
    if illegal:
        raise ValueError(
            "gauge library admits only allowlisted covariant atoms "
            f"{sorted(allow)}; rejected {illegal}"
        )


def _as_float_array(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _validate_connection_arrays(
    A: np.ndarray,
    dA: np.ndarray,
    ddA: np.ndarray,
    *,
    algebra: LieAlgebra,
    signature: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = _as_float_array(A, "A")
    dA = _as_float_array(dA, "dA")
    ddA = _as_float_array(ddA, "ddA")
    if A.ndim != 3:
        raise ValueError(f"A must have shape (B, d, n), got {A.shape}")
    batch, dim, adj = A.shape
    if adj != algebra.dim:
        raise ValueError(
            f"A adjoint dim {adj} != algebra.dim {algebra.dim}"
        )
    if len(signature) != dim:
        raise ValueError(
            f"signature length {len(signature)} != spacetime dim {dim}"
        )
    if any(s not in (-1, 1) for s in signature):
        raise ValueError("signature entries must be +1 or -1")
    if dA.shape != (batch, dim, dim, adj):
        raise ValueError(
            f"dA must have shape {(batch, dim, dim, adj)}, got {dA.shape}"
        )
    if ddA.shape != (batch, dim, dim, dim, adj):
        raise ValueError(
            f"ddA must have shape {(batch, dim, dim, dim, adj)}, got {ddA.shape}"
        )
    return A, dA, ddA


def _eta(signature: tuple[int, ...]) -> np.ndarray:
    return np.asarray(signature, dtype=np.float64)


def _adjoint_action(
    field: np.ndarray, omega: np.ndarray, f: np.ndarray, coupling: float
) -> np.ndarray:
    """Adjoint action ``g f^{pqa} omega^p X^q`` on the last index of ``field``."""
    batch, adj = field.shape[0], field.shape[-1]
    omega_b = np.asarray(omega, dtype=np.float64)
    if omega_b.ndim == 1:
        omega_b = np.broadcast_to(omega_b, (batch, adj))
    if omega_b.shape != (batch, adj):
        raise ValueError(
            f"omega must have shape {(batch, adj)} or {(adj,)}, got {omega_b.shape}"
        )
    flat = field.reshape(batch, -1, adj)
    rotated = coupling * np.einsum("pqa,Bp,Bjq->Bja", f, omega_b, flat)
    return rotated.reshape(field.shape)


def global_gauge_transform_connection(
    A: np.ndarray,
    dA: np.ndarray,
    ddA: np.ndarray,
    U: np.ndarray,
    *,
    algebra: LieAlgebra,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Finite *global* conjugation ``X -> U X U^{-1}`` on ``A``, ``dA``, ``ddA``.

    ``U`` is a constant special-unitary matrix in the fundamental representation.
    """
    gens = algebra.generators()
    u = np.asarray(U)
    u_dag = u.conj().swapaxes(-1, -2)

    def _conj(field: np.ndarray) -> np.ndarray:
        mat = kernels.to_matrix(np, field, gens)
        # field (..., a) -> mat (..., i, j)
        transformed = np.matmul(np.matmul(u, mat), u_dag)
        return np.asarray(kernels.from_matrix(np, transformed, gens), dtype=np.float64)

    return _conj(A), _conj(dA), _conj(ddA)


def random_special_unitary(
    n: int, rng: np.random.Generator
) -> np.ndarray:
    """Haar-ish random ``SU(n)`` matrix (QR of a complex Gaussian, det-fixed)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, _r = np.linalg.qr(z)
    det = np.linalg.det(q)
    q = q / (det ** (1.0 / n))
    return q


@dataclass(frozen=True)
class GaugeCovariantJet:
    r"""First-order gauge-covariant jet: ``F`` and ``D F``, no connection.

    Parameters
    ----------
    F
        Field strength ``F_{mu nu}^a``, shape ``(B, d, d, n)``.
    DF
        Covariant derivative ``(D_rho F_{mu nu})^a``, shape ``(B, d, d, d, n)``.
    ym_eom
        Yang-Mills operator ``(D_mu F^{mu nu})^a``, shape ``(B, d, n)``.
    bianchi
        Bianchi operator ``(D_mu *F^{mu nu})^a``, shape ``(B, d, n)``.
    algebra
        The gauge :class:`LieAlgebra`.
    coupling
        Gauge coupling ``g``.
    signature
        Flat metric signature of length ``d``.
    """

    F: np.ndarray
    DF: np.ndarray
    ym_eom: np.ndarray
    bianchi: np.ndarray
    algebra: LieAlgebra
    coupling: float
    signature: tuple[int, ...]

    @classmethod
    def from_arrays(
        cls,
        A: np.ndarray,
        dA: np.ndarray,
        ddA: np.ndarray,
        *,
        algebra: LieAlgebra,
        coupling: float,
        signature: tuple[int, ...] = EUCLIDEAN_4D,
    ) -> GaugeCovariantJet:
        """Build the jet and drop ``A`` / ``dA`` / ``ddA``."""
        A, dA, ddA = _validate_connection_arrays(
            A, dA, ddA, algebra=algebra, signature=signature
        )
        f = algebra.structure_constants()
        eta = _eta(signature)
        eps = levi_civita_symbol(A.shape[1])
        fld = np.asarray(
            kernels.field_strength(np, A, dA, f, coupling), dtype=np.float64
        )
        cov = np.asarray(
            kernels.covariant_derivative_field_strength(np, A, dA, ddA, f, coupling),
            dtype=np.float64,
        )
        eom = np.asarray(
            kernels.covariant_divergence(np, A, dA, ddA, f, coupling, eta),
            dtype=np.float64,
        )
        bia = np.asarray(
            kernels.bianchi(np, A, dA, ddA, f, coupling, eta, eps),
            dtype=np.float64,
        )
        return cls(
            F=fld,
            DF=cov,
            ym_eom=eom,
            bianchi=bia,
            algebra=algebra,
            coupling=float(coupling),
            signature=tuple(int(s) for s in signature),
        )

    @property
    def batch(self) -> int:
        return int(self.F.shape[0])

    @property
    def spacetime_dim(self) -> int:
        return int(self.F.shape[1])

    def library_names(self) -> tuple[str, ...]:
        return tuple(sorted(LEGAL_SINGLET_ATOMS))

    def singlets(self) -> dict[str, np.ndarray]:
        """Weyl singlets of ``F`` and ``D F``. Keys are :data:`LEGAL_SINGLET_ATOMS`."""
        eta = _eta(self.signature)
        eps = levi_civita_symbol(self.spacetime_dim)
        action = np.asarray(
            kernels.action_density(np, self.F, eta), dtype=np.float64
        )
        topo = np.asarray(
            kernels.topological_charge_density(np, self.F, eps), dtype=np.float64
        )
        ym_sq = np.einsum("n,Bna,Bna->B", eta, self.ym_eom, self.ym_eom)
        bia_sq = np.einsum("n,Bna,Bna->B", eta, self.bianchi, self.bianchi)
        defect = np.asarray(
            kernels.self_duality_defect(np, self.F, eps, eta), dtype=np.float64
        )
        sd_sq = np.einsum("m,n,Bmna,Bmna->B", eta, eta, defect, defect)
        out = {
            SINGLET_TR_F2: np.asarray(action, dtype=np.float64).reshape(-1),
            SINGLET_TR_F_FTILDE: np.asarray(topo, dtype=np.float64).reshape(-1),
            SINGLET_YM_SQ: np.asarray(ym_sq, dtype=np.float64).reshape(-1),
            SINGLET_BIANCHI_SQ: np.asarray(bia_sq, dtype=np.float64).reshape(-1),
            SINGLET_SELF_DUAL_SQ: np.asarray(sd_sq, dtype=np.float64).reshape(-1),
        }
        if set(out) != LEGAL_SINGLET_ATOMS:
            raise RuntimeError("singlet allowlist drifted from singlets()")
        return out

    def adjoint_1forms(self) -> dict[str, np.ndarray]:
        """Adjoint 1-form residuals. Not mixed into singlet STLSQ."""
        return {
            ADJOINT_YM: np.asarray(self.ym_eom, dtype=np.float64),
            ADJOINT_BIANCHI: np.asarray(self.bianchi, dtype=np.float64),
        }


def gauge_equivariance_defect(
    A: np.ndarray,
    dA: np.ndarray,
    ddA: np.ndarray,
    omega: np.ndarray,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...] = EUCLIDEAN_4D,
    eps: float = 1e-4,
    U: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    r"""Singlet defects under an infinitesimal adjoint rotation and a global ``U``.

    Infinitesimal: ``X' = X + eps * g[omega, X]`` (constant ``omega``, so
    ``d omega = 0``) at ``eps`` and ``eps/2``. Singlets must change by
    ``O(eps^2)``; ``F`` itself changes at ``O(eps)`` (homogeneous adjoint).

    Global: finite ``U A U^{-1}``; each singlet is unchanged.
    """
    A, dA, ddA = _validate_connection_arrays(
        A, dA, ddA, algebra=algebra, signature=signature
    )
    f = algebra.structure_constants()
    base = GaugeCovariantJet.from_arrays(
        A, dA, ddA, algebra=algebra, coupling=coupling, signature=signature
    )
    base_s = base.singlets()

    def _shift(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            A + scale * _adjoint_action(A, omega, f, coupling),
            dA + scale * _adjoint_action(dA, omega, f, coupling),
            ddA + scale * _adjoint_action(ddA, omega, f, coupling),
        )

    a1, da1, dda1 = _shift(eps)
    a2, da2, dda2 = _shift(0.5 * eps)
    jet1 = GaugeCovariantJet.from_arrays(
        a1, da1, dda1, algebra=algebra, coupling=coupling, signature=signature
    )
    jet2 = GaugeCovariantJet.from_arrays(
        a2, da2, dda2, algebra=algebra, coupling=coupling, signature=signature
    )
    inf_defects = {
        name: float(np.max(np.abs(jet1.singlets()[name] - base_s[name])))
        for name in LEGAL_SINGLET_ATOMS
    }
    inf_defects_half = {
        name: float(np.max(np.abs(jet2.singlets()[name] - base_s[name])))
        for name in LEGAL_SINGLET_ATOMS
    }
    f_change = float(np.max(np.abs(jet1.F - base.F)))
    f_change_half = float(np.max(np.abs(jet2.F - base.F)))

    if U is None:
        gen = rng if rng is not None else np.random.default_rng(0)
        U = random_special_unitary(algebra.n_fundamental, gen)
    a_g, da_g, dda_g = global_gauge_transform_connection(
        A, dA, ddA, U, algebra=algebra
    )
    jet_g = GaugeCovariantJet.from_arrays(
        a_g, da_g, dda_g, algebra=algebra, coupling=coupling, signature=signature
    )
    global_defects = {
        name: float(np.max(np.abs(jet_g.singlets()[name] - base_s[name])))
        for name in LEGAL_SINGLET_ATOMS
    }
    return {
        "eps": float(eps),
        "infinitesimal": inf_defects,
        "infinitesimal_half": inf_defects_half,
        "F_change": f_change,
        "F_change_half": f_change_half,
        "global": global_defects,
    }


def evaluate_gauge_law_gate(
    equation: Any,
    *,
    lhs_name: str,
    A: np.ndarray,
    dA: np.ndarray,
    ddA: np.ndarray,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...] = EUCLIDEAN_4D,
    extra_columns: Mapping[str, np.ndarray] | None = None,
    U: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Fail-closed equivariance gate for a fitted singlet equation.

    Rebuilds the allowlisted library on a globally conjugated connection and
    requires the residual to be unchanged. Illegal extras are rejected before
    any residual is computed.
    """
    names = list(equation.term_names)
    if extra_columns:
        names = names + list(extra_columns)
    assert_library_gauge_legal(names)
    assert_library_gauge_legal([lhs_name])

    def _library(
        jet: GaugeCovariantJet,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        cols = jet.singlets()
        if extra_columns:
            cols = {**cols, **{k: np.asarray(v, dtype=float).reshape(-1) for k, v in extra_columns.items()}}
        target = cols[lhs_name]
        term_names = list(equation.term_names)
        design = np.stack([cols[name] for name in term_names], axis=1)
        return design, target, term_names

    base = GaugeCovariantJet.from_arrays(
        A, dA, ddA, algebra=algebra, coupling=coupling, signature=signature
    )
    design, target, _terms = _library(base)
    residual = target - equation.predict(design)

    gen = rng if rng is not None else np.random.default_rng(1)
    if U is None:
        U = random_special_unitary(algebra.n_fundamental, gen)
    a_g, da_g, dda_g = global_gauge_transform_connection(
        A, dA, ddA, U, algebra=algebra
    )
    transformed = GaugeCovariantJet.from_arrays(
        a_g, da_g, dda_g, algebra=algebra, coupling=coupling, signature=signature
    )
    design_g, target_g, _ = _library(transformed)
    residual_g = target_g - equation.predict(design_g)
    defect = float(np.max(np.abs(residual_g - residual)))
    singlet_defects = {
        name: float(np.max(np.abs(transformed.singlets()[name] - base.singlets()[name])))
        for name in LEGAL_SINGLET_ATOMS
    }
    passed = bool(defect <= atol and max(singlet_defects.values()) <= atol)
    return {
        "passed": passed,
        "residual_defect": defect,
        "singlet_defects": singlet_defects,
        "atol": float(atol),
        "yang_mills_claim": False,
        "continuum_claim": False,
    }


def covariant_singlet_columns(jet: GaugeCovariantJet) -> dict[str, np.ndarray]:
    """Allowlisted singlet columns. Not a public Yang-Mills discoverer entry."""
    return jet.singlets()


__all__ = [
    "ADJOINT_BIANCHI",
    "ADJOINT_YM",
    "GaugeCovariantJet",
    "LEGAL_ADJOINT_1FORM_ATOMS",
    "LEGAL_SINGLET_ATOMS",
    "SELF_DUAL_ACTION_OVER_TOPOLOGICAL",
    "SINGLET_BIANCHI_SQ",
    "SINGLET_SELF_DUAL_SQ",
    "SINGLET_TR_F2",
    "SINGLET_TR_F_FTILDE",
    "SINGLET_YM_SQ",
    "assert_library_gauge_legal",
    "covariant_singlet_columns",
    "evaluate_gauge_law_gate",
    "gauge_equivariance_defect",
    "global_gauge_transform_connection",
    "random_special_unitary",
]
