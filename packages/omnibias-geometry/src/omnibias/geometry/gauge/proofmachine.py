# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Register the gauge provers behind the :class:`omnibias.core.proof.ProofMachine`.

The machine is the repo's common *prove / disprove* front door: a
:class:`~omnibias.core.proof.Conjecture` goes in and a
:class:`~omnibias.core.proof.Verdict` (``PROVED | BLOCKED``) comes out, with the
schema gate, an independent replay, the honesty gate, and the optional Lean-kernel
gate applied uniformly.

====================================  ====================================================
kind                                  prover
====================================  ====================================================
``transfer_matrix_spectral_gap``      :func:`omnibias.geometry.gauge.transfer.certified_transfer_matrix_gap`
``strong_coupling_glueball_gap``      :func:`omnibias.geometry.gauge.transfer.certified_strong_coupling_glueball_bound`
====================================  ====================================================

The certificate is **sealed**, so:

* the **schema / digest gate** validates the payload and its tamper-evident digest;
* the **replay** rebuilds the transfer matrix from its recorded inputs and re-runs
  the gap engine, rejecting a sealed bound tighter than an independent derivation
  supports;
* the **honesty gate** downgrades any conjecture asserting a continuum or
  Yang-Mills claim, neither of which a fixed-matrix certificate can support;
* the **formal gate** routes the rational ``subdominant_ratio_upper`` obligation to
  the Mathlib-free Lean kernel's ``spectral_gap_pos`` lemma.

``omnibias-pinn`` does not depend on ``omnibias-geometry``, so this prover cannot
join the Birkhoff-Hopf one in :mod:`omnibias.pinn.certified.machine`.  Owning a
registry per package is the established pattern -- see
:mod:`omnibias.sos.proofmachine`.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from omnibias.core.proof import (
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
)
from omnibias.geometry.gauge.transfer.certificates import (
    STRONG_COUPLING_KIND,
    TRANSFER_GAP_KIND,
    replay_strong_coupling_gap,
    replay_transfer_matrix_gap,
    seal_strong_coupling_certificate,
    seal_transfer_gap_certificate,
    strong_coupling_schema_errors,
    transfer_gap_schema_errors,
)
from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
from omnibias.geometry.gauge.transfer.matrices import rebuild
from omnibias.geometry.gauge.transfer.strong_coupling import (
    certified_strong_coupling_glueball_bound,
)


def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _prove_transfer_matrix_gap(conjecture: Conjecture) -> ProofAttempt:
    """Build the transfer matrix named by the conjecture and certify its gap."""
    data: Mapping[str, Any] = conjecture.data
    parameters = data.get("parameters")
    if not isinstance(parameters, Mapping):
        return _blocked(
            "conjecture data must carry a 'parameters' mapping naming a transfer-"
            "matrix builder and its arguments"
        )
    try:
        transfer = rebuild(parameters)
    except (ValueError, TypeError, KeyError) as exc:
        return _blocked(f"could not build transfer matrix: {exc}")
    try:
        result = certified_transfer_matrix_gap(
            transfer,
            lattice_spacing=data.get("lattice_spacing"),
            deflate=bool(data.get("deflate", True)),
        )
    except (ValueError, TypeError) as exc:
        return _blocked(f"could not certify a spectral gap: {exc}")
    if not result.certified:
        return _blocked(
            f"no positive gap certified for {result.model} "
            f"(subdominant ratio bound {result.subdominant_ratio_upper:.6g} >= 1)"
        )
    minimum = data.get("min_spectral_gap")
    if isinstance(minimum, int | float) and result.spectral_gap_lower < float(minimum):
        return _blocked(
            f"certified gap {result.spectral_gap_lower:.6g} is below the requested "
            f"threshold {float(minimum):.6g}"
        )
    sealed = seal_transfer_gap_certificate(result, transfer, claim=conjecture.name)
    detail = (
        f"{result.model} ({result.basis} basis, dim {result.dimension}): "
        f"m a >= {result.spectral_gap_lower:.6g} via {result.method}"
    )
    return ProofAttempt(status="PROVED", certificate=sealed, detail=detail)


def _prove_strong_coupling_gap(conjecture: Conjecture) -> ProofAttempt:
    """Certify the crude polymer bound named by the conjecture."""
    data: Mapping[str, Any] = conjecture.data
    beta = data.get("beta")
    if isinstance(beta, bool) or not isinstance(beta, int | float | Fraction) or float(beta) <= 0.0:
        return _blocked("conjecture data must carry a positive numeric 'beta'")
    dim = data.get("spacetime_dim", 4)
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 2:
        return _blocked("spacetime_dim must be an integer >= 2")
    try:
        result = certified_strong_coupling_glueball_bound(beta, spacetime_dim=dim)
    except (ValueError, TypeError) as exc:
        return _blocked(f"could not certify a polymer bound: {exc}")
    if not result.certified:
        return _blocked(
            f"no certified polymer bound at beta={float(beta):.6g} "
            f"(C u bound {result.subdominant_ratio_upper:.6g} >= 1)"
        )
    minimum = data.get("min_spectral_gap")
    if isinstance(minimum, int | float) and result.spectral_gap_lower < float(minimum):
        return _blocked(
            f"certified gap {result.spectral_gap_lower:.6g} is below the requested "
            f"threshold {float(minimum):.6g}"
        )
    sealed = seal_strong_coupling_certificate(result, claim=conjecture.name)
    detail = (
        f"su2_wilson_polymer (d={result.spacetime_dim}, C={result.coordination}): "
        f"m a >= {result.spectral_gap_lower:.6g} via {result.method}"
    )
    return ProofAttempt(status="PROVED", certificate=sealed, detail=detail)


def gauge_provers() -> list[FunctionProver]:
    """The gauge provers registered by :func:`build_gauge_machine`."""
    return [
        FunctionProver(
            name="transfer_matrix_spectral_gap",
            kinds=frozenset({TRANSFER_GAP_KIND}),
            prove_fn=_prove_transfer_matrix_gap,
            schema_fn=transfer_gap_schema_errors,
            replay_fn=replay_transfer_matrix_gap,
        ),
        FunctionProver(
            name="strong_coupling_glueball_gap",
            kinds=frozenset({STRONG_COUPLING_KIND}),
            prove_fn=_prove_strong_coupling_gap,
            schema_fn=strong_coupling_schema_errors,
            replay_fn=replay_strong_coupling_gap,
        ),
    ]


def build_gauge_machine() -> ProofMachine:
    """A :class:`~omnibias.core.proof.ProofMachine` preloaded with the gauge provers."""
    machine = ProofMachine()
    for prover in gauge_provers():
        machine.register(prover)
    return machine


__all__ = [
    "STRONG_COUPLING_KIND",
    "TRANSFER_GAP_KIND",
    "build_gauge_machine",
    "gauge_provers",
]
