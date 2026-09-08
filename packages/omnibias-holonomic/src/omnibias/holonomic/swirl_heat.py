# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""D-finite swirl-heat identity (theory 07-12).

In the similarity coordinate ``xi = 1/r``, the isotropic exterior
``K = 1/r = xi`` is annihilated by ``-xi d/dxi + 1``. That is a
polynomial identity over ``Q``, accepted only via
:func:`~omnibias.core.proof.engine.prove` on ``identity``. Not a 3-D
heat theorem and not a Navier–Stokes claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from omnibias.core.proof.engine import prove
from omnibias.core.verified.swirl_heat import honesty_payload

SWIRL_HEAT_KIND = "swirl_heat_identity"
_DOMAIN = [1.0, 2.0]


def named_swirl_heat_generators() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``K = xi`` and the annihilator polynomial ``-xi K' + K``.

    ``K`` coefficients ``(0, 1)``. ``K' = 1``, so ``-xi K' + K = 0``.
    """
    return (0, 1), (0,)


def swirl_heat_identity_payload(
    generators: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    _k, annihilator = generators or named_swirl_heat_generators()
    return {
        "left": list(annihilator),
        "right": [0],
        "domain": list(_DOMAIN),
        "kind": SWIRL_HEAT_KIND,
        "honesty": honesty_payload(),
    }


def prove_swirl_heat_identity(
    generators: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    *,
    lean_check: bool = False,
) -> Any:
    """Dispatch the annihilator through ``prove("identity")``."""
    payload = swirl_heat_identity_payload(generators)
    return prove("identity", payload, name=SWIRL_HEAT_KIND, lean_check=lean_check)


def replay_swirl_heat_identity(certificate: Mapping[str, Any]) -> bool | None:
    payload = certificate.get("payload") if isinstance(certificate, Mapping) else None
    if not isinstance(payload, Mapping):
        data = certificate if isinstance(certificate, Mapping) else None
        if not isinstance(data, Mapping):
            return None
        if data.get("kind") != SWIRL_HEAT_KIND and data.get("left") is None:
            return None
        payload = data
    left = payload.get("left")
    right = payload.get("right")
    if not isinstance(left, Sequence) or not isinstance(right, Sequence):
        return False
    return list(left) == [0] and list(right) in ([0], [0.0])


__all__ = [
    "SWIRL_HEAT_KIND",
    "named_swirl_heat_generators",
    "prove_swirl_heat_identity",
    "replay_swirl_heat_identity",
    "swirl_heat_identity_payload",
]
