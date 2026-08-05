# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge connection schema (pure Python).

A :class:`GaugeConnectionSpec` bundles a :class:`LieAlgebra`, the coupling ``g``,
the spacetime dimension and metric signature, and the degree-1
:class:`LieAlgebraValuedForm` whose components name the connection fields
``A_mu^a`` inside a :class:`FieldState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibias.geometry.gauge._core.forms import LieAlgebraValuedForm
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra

EUCLIDEAN_4D: tuple[int, ...] = (1, 1, 1, 1)
MINKOWSKI_4D: tuple[int, ...] = (-1, 1, 1, 1)


@dataclass(frozen=True)
class GaugeConnectionSpec:
    """A gauge connection ``A_mu^a`` on flat ``R^d``.

    Parameters
    ----------
    algebra
        The gauge :class:`LieAlgebra`.
    form
        Degree-1 :class:`LieAlgebraValuedForm` mapping ``(mu,)`` to the tuple of
        component names ``A_mu^a`` (one per generator).
    coupling
        The gauge coupling ``g``.
    signature
        Flat metric signature, e.g. ``(1, 1, 1, 1)`` (Euclidean) or
        ``(-1, 1, 1, 1)`` (Minkowski). Length must equal ``spacetime_dim``.
    gauge
        Gauge-fixing label (metadata only), e.g. ``"lorenz"``.
    """

    algebra: LieAlgebra
    form: LieAlgebraValuedForm
    coupling: float = 1.0
    signature: tuple[int, ...] = field(default=EUCLIDEAN_4D)
    gauge: str = "lorenz"

    def __post_init__(self) -> None:
        if self.form.degree != 1:
            raise ValueError("a connection form must have degree 1")
        if self.form.adjoint_dim != self.algebra.dim:
            raise ValueError(
                f"form adjoint_dim {self.form.adjoint_dim} != algebra dim {self.algebra.dim}"
            )
        if len(self.signature) != self.spacetime_dim:
            raise ValueError(
                f"signature length {len(self.signature)} != spacetime_dim {self.spacetime_dim}"
            )
        if any(s not in (-1, 1) for s in self.signature):
            raise ValueError("signature entries must be +1 or -1")

    @property
    def spacetime_dim(self) -> int:
        return self.form.dim

    def component_name(self, mu: int, a: int) -> str | None:
        names = self.form.comps.get((mu,))
        return None if names is None else names[a]


def gauge_connection_spec(
    algebra: LieAlgebra,
    *,
    coupling: float = 1.0,
    spacetime_dim: int = 4,
    signature: tuple[int, ...] = EUCLIDEAN_4D,
    name: str = "A",
    gauge: str = "lorenz",
) -> GaugeConnectionSpec:
    """Build a connection spec with auto-named components ``{name}_{mu}_{a}``."""
    comps = {
        (mu,): tuple(f"{name}_{mu}_{a}" for a in range(algebra.dim))
        for mu in range(spacetime_dim)
    }
    form = LieAlgebraValuedForm(
        degree=1, dim=spacetime_dim, adjoint_dim=algebra.dim, comps=comps
    )
    return GaugeConnectionSpec(
        algebra=algebra, form=form, coupling=coupling, signature=signature, gauge=gauge
    )


def connection_component_names(spec: GaugeConnectionSpec) -> tuple[str, ...]:
    """All component names in ``(mu, a)`` row-major order."""
    out: list[str] = []
    for mu in range(spec.spacetime_dim):
        names = spec.form.comps.get((mu,))
        if names is None:
            raise ValueError(f"connection form missing component for axis {mu}")
        out.extend(names)
    return tuple(out)


__all__ = [
    "EUCLIDEAN_4D",
    "GaugeConnectionSpec",
    "MINKOWSKI_4D",
    "connection_component_names",
    "gauge_connection_spec",
]
