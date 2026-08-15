# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Wilson / Polyakov loop language (the language trap).

A finite-order jet of ``A`` -- even the covariant jet of ``F`` and ``D F`` --
is a pointwise Taylor chart. Path-ordered holonomy around a finite cycle is a
different object. This module evaluates the legal loop atoms on lattice links
and refuses to treat a jet as a loop source (or a Green function as a jet atom).

Honesty: classical gauge-invariant traces on a finite lattice, or a planted
area-law identity. Not a continuum string tension, not a Yang-Mills mass gap,
not quantum Yang-Mills. GEVP / transfer-gap certificates are not search atoms.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, NoReturn

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge.lattice._core.kernels import (
    gauge_transform_links,
    plaquette_trace,
    polyakov_loop_field,
    wilson_loop_trace,
)
from omnibias.geometry.gauge.lattice._core.stats import creutz_ratio_value

LOOP_PLAQUETTE = "plaquette"
LOOP_W11 = "W(1,1)"
LOOP_W21 = "W(2,1)"
LOOP_W12 = "W(1,2)"
LOOP_W22 = "W(2,2)"
LOOP_POLYAKOV = "Polyakov"

LEGAL_LOOP_ATOMS: frozenset[str] = frozenset(
    {
        LOOP_PLAQUETTE,
        LOOP_W11,
        LOOP_W21,
        LOOP_W12,
        LOOP_W22,
        LOOP_POLYAKOV,
    }
)

_WILSON_NAME = re.compile(r"^W\((\d+),(\d+)\)$")
_CERT_ATOM_NAMES: frozenset[str] = frozenset(
    {
        "gevp",
        "gevp_plateau",
        "transfer_gap",
        "glueball_mass",
        "certified_transfer_matrix_gap",
    }
)

WILSON_EXTENTS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2))
LOOP_INVARIANCE_ATOL = 1e-10


def is_loop_atom_name(name: str) -> bool:
    """True for Wilson / Polyakov / plaquette names (legal or not)."""
    if name in LEGAL_LOOP_ATOMS or name in _CERT_ATOM_NAMES:
        return True
    key = name.lower()
    return (
        key in {"wilson", "polyakov", "plaquette"}
        or "wilson" in key
        or "polyakov" in key
        or _WILSON_NAME.match(name) is not None
    )


def is_green_column_name(name: str) -> bool:
    """True for inverse-Laplacian / Green names that must not enter a jet."""
    key = name.lower().replace(" ", "")
    return (
        "inverse_laplacian" in key
        or key.startswith("green(")
        or "delta^{-1}" in key
        or "delta^-1" in key
        or key in {"green", "inv_lap", "inverse-laplacian"}
    )


def refuse_jet_as_loop_source(
    obj: object | None = None, *_args: object, **_kwargs: object
) -> NoReturn:
    """A covariant jet is not a Wilson / Polyakov source."""
    _ = obj
    raise ValueError(
        "a gauge-covariant jet is not a loop source (language split); "
        "Wilson/Polyakov holonomy is evaluated on lattice links, not F/DF"
    )


def refuse_loop_as_covariant_jet(
    names: Iterable[str] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """Loop traces are not a covariant jet and not STLSQ extras on one."""
    extra = f"; rejected {list(names)}" if names is not None else ""
    raise ValueError(
        "loop traces are not a covariant jet (language split); "
        "never form partial^k A or F from Wilson/Polyakov holonomy"
        f"{extra}"
    )


def refuse_green_as_jet_atom(
    names: Iterable[str] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """An inverse Laplacian is not a jet atom and not a holonomy."""
    extra = f"; rejected {list(names)}" if names is not None else ""
    raise ValueError(
        "an inverse Laplacian / Green function is not a jet atom "
        "and is not a Wilson holonomy"
        f"{extra}"
    )


def assert_library_loop_legal(
    names: Iterable[str],
    *,
    allow: frozenset[str] = LEGAL_LOOP_ATOMS,
) -> None:
    """Raise ``ValueError`` if any name is outside the closed loop allowlist."""
    illegal = [name for name in names if name not in allow]
    if illegal:
        raise ValueError(
            "loop library admits only allowlisted Wilson/Polyakov atoms "
            f"{sorted(allow)}; rejected {illegal}"
        )


@dataclass(frozen=True)
class LoopObservableTable:
    """Gauge-invariant loop traces. Not a jet and not a continuum connection."""

    values: dict[str, np.ndarray]
    source: Literal["lattice_links"] = "lattice_links"


def _as_link_array(field: LatticeLinkField) -> np.ndarray:
    links = np.asarray(field.links, dtype=np.float64)
    if links.ndim != 6 or links.shape[0] != 4 or links.shape[-1] != 4:
        raise ValueError(f"links must have shape (4, *L4, 4), got {links.shape}")
    if len(links.shape) - 2 != 4:
        raise ValueError(f"lattice must be 4-D, got shape {links.shape}")
    return links


def _oriented_planes() -> tuple[tuple[int, int], ...]:
    return tuple((mu, nu) for mu in range(4) for nu in range(mu + 1, 4))


def _mean_plaquette(links: np.ndarray) -> float:
    parts = [
        float(np.mean(np.asarray(plaquette_trace(np, links, mu, nu), dtype=np.float64)))
        for mu, nu in _oriented_planes()
    ]
    return float(np.mean(parts))


def _mean_wilson(links: np.ndarray, r_extent: int, t_extent: int) -> float:
    parts = [
        float(
            np.mean(
                np.asarray(
                    wilson_loop_trace(np, links, mu, r_extent, t_extent, t_dir=nu),
                    dtype=np.float64,
                )
            )
        )
        for mu, nu in _oriented_planes()
    ]
    return float(np.mean(parts))


def wilson_plaquette_pairs(field: LatticeLinkField) -> tuple[np.ndarray, np.ndarray]:
    """Sitewise ``W(1,1)`` and plaquette on every oriented plane (same sites)."""
    links = _as_link_array(field)
    wilson: list[np.ndarray] = []
    plaquette: list[np.ndarray] = []
    for mu, nu in _oriented_planes():
        wilson.append(
            np.asarray(
                wilson_loop_trace(np, links, mu, 1, 1, t_dir=nu), dtype=np.float64
            ).reshape(-1)
        )
        plaquette.append(
            np.asarray(plaquette_trace(np, links, mu, nu), dtype=np.float64).reshape(-1)
        )
    return np.concatenate(wilson), np.concatenate(plaquette)


def evaluate_loop_atoms(field: LatticeLinkField) -> LoopObservableTable:
    """Volume-mean loop atoms. Creutz / GEVP / transfer-gap are not included."""
    links = _as_link_array(field)
    values = {
        LOOP_PLAQUETTE: np.asarray([_mean_plaquette(links)], dtype=np.float64),
        LOOP_W11: np.asarray([_mean_wilson(links, 1, 1)], dtype=np.float64),
        LOOP_W21: np.asarray([_mean_wilson(links, 2, 1)], dtype=np.float64),
        LOOP_W12: np.asarray([_mean_wilson(links, 1, 2)], dtype=np.float64),
        LOOP_W22: np.asarray([_mean_wilson(links, 2, 2)], dtype=np.float64),
        LOOP_POLYAKOV: np.asarray(
            [float(np.mean(np.asarray(polyakov_loop_field(np, links), dtype=np.float64)))],
            dtype=np.float64,
        ),
    }
    return LoopObservableTable(values=values, source="lattice_links")


def creutz_ratio_from_wilson(
    values: Mapping[str, np.ndarray] | LoopObservableTable,
    *,
    r_extent: int = 2,
    t_extent: int = 2,
) -> float:
    """Derived Creutz identity on Wilson traces. Not a search atom."""
    table = values.values if isinstance(values, LoopObservableTable) else values
    w_rt = float(np.mean(np.asarray(table[f"W({r_extent},{t_extent})"], dtype=np.float64)))
    w_r1t1 = float(
        np.mean(np.asarray(table[f"W({r_extent - 1},{t_extent - 1})"], dtype=np.float64))
    )
    w_r1t = float(
        np.mean(np.asarray(table[f"W({r_extent - 1},{t_extent})"], dtype=np.float64))
    )
    w_rt1 = float(
        np.mean(np.asarray(table[f"W({r_extent},{t_extent - 1})"], dtype=np.float64))
    )
    return float(creutz_ratio_value(w_rt, w_r1t1, w_r1t, w_rt1))


def evaluate_loop_gauge_invariance(
    field: LatticeLinkField,
    g: np.ndarray,
    *,
    atol: float = LOOP_INVARIANCE_ATOL,
) -> dict[str, object]:
    """Fail-closed loop-trace invariance. Not a mass-gap claim."""
    links = _as_link_array(field)
    gauge = np.asarray(g, dtype=np.float64)
    if gauge.shape != (*links.shape[1:-1], 4):
        raise ValueError(
            f"g must have shape {(*links.shape[1:-1], 4)}, got {gauge.shape}"
        )
    transformed = LatticeLinkField(
        links=np.asarray(gauge_transform_links(np, links, gauge), dtype=np.float64),
        spacing=field.spacing,
    )
    base = evaluate_loop_atoms(field)
    other = evaluate_loop_atoms(transformed)
    defects = [
        abs(float(np.mean(base.values[name])) - float(np.mean(other.values[name])))
        for name in sorted(LEGAL_LOOP_ATOMS)
    ]
    max_abs = float(max(defects))
    return {
        "passed": bool(max_abs <= atol),
        "max_abs": max_abs,
        "atol": float(atol),
        "yang_mills_claim": False,
        "continuum_claim": False,
    }


def identity_numpy_links(lattice_shape: tuple[int, int, int, int]) -> np.ndarray:
    """Cold-start unit quaternions, shape ``(4, *lattice_shape, 4)``."""
    links = np.zeros((4, *lattice_shape, 4), dtype=np.float64)
    links[..., 0] = 1.0
    return links


def random_numpy_links(
    lattice_shape: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Haar-ish random unit-quaternion links (independent sites / directions)."""
    raw = rng.normal(size=(4, *lattice_shape, 4))
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / np.maximum(norms, 1e-30)


__all__ = [
    "LEGAL_LOOP_ATOMS",
    "LOOP_INVARIANCE_ATOL",
    "LOOP_PLAQUETTE",
    "LOOP_POLYAKOV",
    "LOOP_W11",
    "LOOP_W12",
    "LOOP_W21",
    "LOOP_W22",
    "LoopObservableTable",
    "WILSON_EXTENTS",
    "assert_library_loop_legal",
    "creutz_ratio_from_wilson",
    "evaluate_loop_atoms",
    "evaluate_loop_gauge_invariance",
    "identity_numpy_links",
    "is_green_column_name",
    "is_loop_atom_name",
    "random_numpy_links",
    "refuse_green_as_jet_atom",
    "refuse_jet_as_loop_source",
    "refuse_loop_as_covariant_jet",
    "wilson_plaquette_pairs",
]
