# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Legal data paths for a gauge-covariant jet (the data trap).

High-order finite differences on a lattice or mesh are a high-pass filter.
Closed-form ``sigma^{(n)}`` on a random-feature interpolant is exact for the
interpolant, not for the gauge field. This module names the legal sources and
refuses the illegal ones.

Legal
-----
- ``analytic`` / ``spectral``: closed-form ``A, dA, ddA`` (path A).
- lattice **links**: plaquettes / APE, never ``partial^k A`` (path C).

Illegal
-------
- lattice or mesh links interpolated by a random-feature field, then jetted
  (path D).
- treating 1-D Fredholm / Volterra / running integrals as the 4-D Yang-Mills
  weak form.

Honesty: a typical Monte Carlo vacuum configuration does not satisfy
``D*F = 0``. Denoising cannot invent a classical EOM. Not a mass-gap claim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, NoReturn

ConnectionSource = Literal[
    "analytic",
    "spectral",
    "lattice_links",
    "random_feature",
    "finite_difference",
]


@dataclass(frozen=True)
class LatticeLinkField:
    """SU(2) link field. Not a jet and not a continuum connection.

    ``links`` has the lattice-kernel shape ``(4, *lattice_shape, 4)``
    (unit quaternions). Spacing is recorded only; it is never used to form
    ``partial^k A``.
    """

    links: object
    spacing: float = 1.0


def refuse_connection_jet_from_links(
    links: object | None = None, *_args: object, **_kwargs: object
) -> NoReturn:
    """Never form ``partial^k A`` from lattice links."""
    _ = links
    raise ValueError(
        "never form a connection jet from lattice links (path C); "
        "the legal local atom is a plaquette, not partial^k A"
    )


def refuse_lattice_random_feature_jet(
    *_args: object, **_kwargs: object
) -> NoReturn:
    """Path D: lattice / mesh links interpolated by a random-feature field."""
    raise ValueError(
        "refusing a random-feature interpolation of lattice links "
        "(path D): closed-form sigma^(n) of an interpolant is not a "
        "gauge-field jet"
    )


def refuse_scalar_integral_as_ym_weak_form(
    names: Iterable[str] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """1-D Fredholm / Volterra / running integrals are not 4-D Yang-Mills."""
    extra = f"; rejected {list(names)}" if names is not None else ""
    raise ValueError(
        "1-D Fredholm/Volterra/running integral columns are not the "
        "4-D Yang-Mills weak form"
        f"{extra}"
    )


def is_scalar_integral_column_name(name: str) -> bool:
    """True for 1-D integral feature names that must not enter YM STLSQ."""
    key = name.lower()
    return (
        "fredholm" in key
        or "volterra" in key
        or key.startswith("i(")
        or "running_integral" in key
        or key in {"i(v)", "integral_running"}
    )


__all__ = [
    "ConnectionSource",
    "LatticeLinkField",
    "is_scalar_integral_column_name",
    "refuse_connection_jet_from_links",
    "refuse_lattice_random_feature_jet",
    "refuse_scalar_integral_as_ym_weak_form",
]
