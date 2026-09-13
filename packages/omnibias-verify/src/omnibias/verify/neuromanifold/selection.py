# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Temperature relaxation and certified decoding of a finite proposal catalogue."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from omnibias.core.realization.transition import TransitionProposal
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv

from .confluence import ConfluenceCertificate, replay_confluence_certificate

T = TypeVar("T")


@dataclass(frozen=True)
class ArchitectureSelection(Generic[T]):
    weights: T
    soft_minimum: T
    gap: Interval
    candidate_ids: tuple[str, ...]
    scope: str = "fixed_finite_catalogue_costs"


def _gap(ids: tuple[str, ...], beta: float) -> Interval:
    if not ids or len(set(ids)) != len(ids) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("unique nonempty candidate IDs and finite positive beta required")
    if len(ids) == 1:
        return Interval.point(0)
    return ln_iv(Interval.from_rational(len(ids))) / Interval.point(beta)


def select_architectures_torch(
    costs: T, candidate_ids: tuple[str, ...], *, beta: float = 1.0
) -> ArchitectureSelection[T]:
    """soft_min <= min(costs) <= soft_min+log(N)/beta, using the shipped tower.

    Costs stay live. This finite-catalogue gap says nothing about retraining or
    unseen architectures; each decoded structural commit needs its own proof.
    """
    from typing import cast

    from omnibias.struct.torch import logsumexp_beta, softmax_beta
    from torch import Tensor

    values = cast(Tensor, costs)
    gap = _gap(candidate_ids, beta)
    if values.ndim != 1 or values.shape[0] != len(candidate_ids):
        raise ValueError("one scalar cost per candidate required")
    return ArchitectureSelection(
        cast(T, softmax_beta(-values, beta)),
        cast(T, -logsumexp_beta(-values, beta)),
        gap,
        candidate_ids,
    )


def select_architectures_jax(
    costs: T, candidate_ids: tuple[str, ...], *, beta: float = 1.0
) -> ArchitectureSelection[T]:
    """JAX twin of the finite temperature-relaxed architecture catalogue."""
    from typing import cast

    from jax import Array
    from omnibias.struct.jax import logsumexp_beta, softmax_beta

    values = cast(Array, costs)
    gap = _gap(candidate_ids, beta)
    if values.ndim != 1 or values.shape[0] != len(candidate_ids):
        raise ValueError("one scalar cost per candidate required")
    return ArchitectureSelection(
        cast(T, softmax_beta(-values, beta)),
        cast(T, -logsumexp_beta(-values, beta)),
        gap,
        candidate_ids,
    )


def decode_architecture(
    proposals: Sequence[TransitionProposal],
    costs: Sequence[float],
    *,
    certify: Callable[[TransitionProposal], ConfluenceCertificate],
) -> tuple[TransitionProposal, ConfluenceCertificate] | None:
    """Choose the cheapest actually certified proposal, with stable slot-ID ties."""
    if len(proposals) != len(costs) or any(not math.isfinite(x) for x in costs):
        raise ValueError("one finite cost per proposal required")
    ranked = sorted(
        zip(costs, proposals, strict=True), key=lambda item: (item[0], item[1].slot_ids)
    )
    for _, proposal in ranked:
        certificate = certify(proposal)
        if (
            certificate.accepted
            and certificate.source_digest == proposal.source_digest
            and certificate.budget == proposal.error_budget
            and certificate.certificate["meta"].get("proposal") == proposal.to_json()
            and replay_confluence_certificate(certificate.certificate)
        ):
            return proposal, certificate
    return None


__all__ = [
    "ArchitectureSelection",
    "decode_architecture",
    "select_architectures_jax",
    "select_architectures_torch",
]
