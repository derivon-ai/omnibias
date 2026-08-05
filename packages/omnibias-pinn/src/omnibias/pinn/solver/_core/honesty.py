# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Honesty labels for omnibias.pinn.solver.

Every operation in this package is labelled honestly as one of:

* ``CLOSED_FORM`` -- the exact ``sigma``-tower derivative path (a single
  ``sigma`` evaluation per order, no nested autodiff);
* ``AUTODIFF`` -- backend automatic differentiation (used for *parameter*
  gradients in the residual-minimisation solver);
* ``NUMERICAL`` -- a standard numerical scheme (e.g. an RK4 or implicit
  time step, a least-squares linear solve);
* ``SPECTRAL`` -- FFT-based spatial derivatives on a periodic grid;
* ``HIGH_ORDER`` -- a high-order Taylor / jet scheme (local truncation
  ``O(dt^{N+1})``), exact only in the truncation limit.

The certificate-style boolean flags mirror
:func:`omnibias.core.proof.certificate.make_certificate`: ``unproven_claim`` is
**hard-wired to ``False``** and :func:`honesty_labels` refuses to emit
``unproven_claim=True``.
"""

from __future__ import annotations

from typing import Any

CLOSED_FORM = "closed-form"
AUTODIFF = "autodiff"
NUMERICAL = "numerical"
SPECTRAL = "spectral"
HIGH_ORDER = "high-order"

#: The method labels this package uses to describe an operation honestly.
METHOD_LABELS = frozenset({CLOSED_FORM, AUTODIFF, NUMERICAL, SPECTRAL, HIGH_ORDER})


def honesty_labels(
    *,
    interval_verified: bool = False,
    continuum_claim: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Return a certificate-style honesty dict with ``unproven_claim`` fixed False.

    Reserved formal flags such as ``theorem_prover_verified`` are *not* included:
    they are earned only by the Lean kernel / ProofMachine and must never be
    stamped into a sealed honesty body.

    Raises
    ------
    ValueError
        If any caller tries to set ``unproven_claim`` (or ``continuum_claim``)
        truthy. omnibias.pinn.solver never asserts a global-regularity / continuum claim.
        Also raised if a reserved honesty key is supplied via ``extra``.
    """
    from omnibias.core.proof.certificate import RESERVED_HONESTY_KEYS

    labels: dict[str, Any] = {
        "unproven_claim": False,
        "continuum_claim": bool(continuum_claim),
        "interval_verified": bool(interval_verified),
    }
    for key, value in extra.items():
        if key in RESERVED_HONESTY_KEYS:
            raise ValueError(
                f"honesty must not contain reserved key {key!r}; it is earned "
                "only by the Lean kernel / ProofMachine"
            )
        labels[key] = value
    assert_no_unproven_claim(labels)
    if labels["continuum_claim"]:
        raise ValueError(
            "omnibias.pinn.solver solves discretised problems and never asserts a "
            "continuum-regularity claim; continuum_claim must be False"
        )
    return labels


def assert_no_unproven_claim(labels: dict[str, Any]) -> None:
    """Guard: raise if ``labels`` asserts ``unproven_claim=True``."""
    if bool(labels.get("unproven_claim", False)):
        raise ValueError(
            "omnibias.pinn.solver never asserts unproven_claim=True; this is a hard "
            "invariant of the package (see the package non-goals)"
        )


__all__ = [
    "AUTODIFF",
    "CLOSED_FORM",
    "HIGH_ORDER",
    "METHOD_LABELS",
    "NUMERICAL",
    "SPECTRAL",
    "assert_no_unproven_claim",
    "honesty_labels",
]
