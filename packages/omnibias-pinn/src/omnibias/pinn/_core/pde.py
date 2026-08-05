# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic PDE / equation schemas.

Each backend's ``equations/`` subpackage builds on these schemas; they
are pure Python (no tensor lib) so the user-facing equation classes can
share signatures and metadata across torch / jax.

Three artefacts live here:

- :class:`ResidualPolicy` -- options describing how a residual should be
  reduced into a scalar loss (mean / Sobolev / WP-causal). Equation
  classes take this as configuration.
- :class:`IncompressibilityPolicy` -- ``"soft"`` vs ``"hard"`` plus a
  scaling weight when soft. Used by the NS equation and by the
  ``cage.IncompressibleProjection`` wrapper.
- :class:`EquationSpec` -- a small descriptor holding the equation name,
  the component groups it expects on the state, and a free-form ``meta``
  dict for diagnostics (units, etc.). Used by the equation registry to
  validate that a given ``FieldState`` matches the equation's
  expectations *before* invoking the residual builder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ResidualPolicy:
    """How to reduce a vector residual into a scalar loss.

    Parameters
    ----------
    reduction
        ``"mean"`` (vanilla MSE), ``"sobolev"`` (Sobolev preconditioned
        on the Fourier basis), or ``"wp_causal"`` (Wang & Perdikaris
        2022 time-causal). Default ``"mean"``.
    sobolev_p
        Exponent for the Sobolev preconditioner (``H^{-p}`` of the
        residual). Used only when ``reduction == "sobolev"``.
    causal_epsilon
        Causal scale in the WP weighting. Used only when
        ``reduction == "wp_causal"``.
    causal_n_time_bins
        Number of time bins in the WP weighting.
    """

    reduction: Literal["mean", "sobolev", "wp_causal"] = "mean"
    sobolev_p: float = 1.0
    causal_epsilon: float = 1.0
    causal_n_time_bins: int = 32


@dataclass(frozen=True)
class IncompressibilityPolicy:
    """How to enforce ``nabla . u = 0``.

    Parameters
    ----------
    mode
        ``"soft"`` -- add a continuity penalty to the loss with weight
        :attr:`weight`. ``"hard"`` -- assume the field has been wrapped
        in :class:`cage.IncompressibleProjection` and skip the penalty.
    weight
        Penalty coefficient when ``mode == "soft"``. Default 1.0.
    """

    mode: Literal["soft", "hard"] = "soft"
    weight: float = 1.0


@dataclass(frozen=True)
class EquationSpec:
    """Metadata describing one PDE residual.

    Parameters
    ----------
    name
        Unique key in the equation registry (e.g. ``"navier_stokes"``).
    required_components
        Component names the equation needs on the state (e.g.
        ``("u", "v", "w", "p")`` for primitive 3D NS).
    required_groups
        Component-group names the equation needs (e.g. ``("velocity",)``).
        These are validated against ``ComponentSpec.groups`` rather than
        ``ComponentSpec.names``.
    requires_time
        ``True`` if the equation is time-dependent. Will be cross-checked
        against ``CoordinateSpec.time_axis`` at residual-build time.
    meta
        Free-form metadata for diagnostics / serialisation.

    Notes
    -----
    The spec does *not* know how to build the residual; the equation
    class in ``omnibias.pinn.{torch,jax}.equations`` does. The spec is
    pure configuration so cross-backend tests can compare equation
    signatures without importing torch.
    """

    name: str
    required_components: tuple[str, ...] = ()
    required_groups: tuple[str, ...] = ()
    requires_time: bool = True
    meta: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        name: str,
        *,
        required_components: Sequence[str] = (),
        required_groups: Sequence[str] = (),
        requires_time: bool = True,
        meta: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("EquationSpec.name must be a non-empty string")
        rc = tuple(required_components)
        rg = tuple(required_groups)
        for c in rc:
            if not isinstance(c, str) or not c:
                raise ValueError(
                    f"required_components must be non-empty strings, got {c!r}"
                )
        for g in rg:
            if not isinstance(g, str) or not g:
                raise ValueError(
                    f"required_groups must be non-empty strings, got {g!r}"
                )
        m = tuple(sorted((meta or {}).items()))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "required_components", rc)
        object.__setattr__(self, "required_groups", rg)
        object.__setattr__(self, "requires_time", bool(requires_time))
        object.__setattr__(self, "meta", m)

    def validate_state(self, state) -> None:
        """Raise :class:`ValueError` if the state does not match the spec."""
        comps = state.components
        for c in self.required_components:
            if not comps.is_component(c):
                raise ValueError(
                    f"equation {self.name!r} requires component {c!r}; "
                    f"state has {comps.names!r}"
                )
        for g in self.required_groups:
            if not comps.is_group(g):
                raise ValueError(
                    f"equation {self.name!r} requires group {g!r}; "
                    f"state has groups {tuple(gn for gn, _ in comps.groups)!r}"
                )
        if self.requires_time and state.coordinate_spec.time_axis is None:
            raise ValueError(
                f"equation {self.name!r} is time-dependent but state's "
                f"coordinate spec has no time axis ({state.coordinate_spec!r})"
            )


__all__ = [
    "EquationSpec",
    "IncompressibilityPolicy",
    "ResidualPolicy",
]
