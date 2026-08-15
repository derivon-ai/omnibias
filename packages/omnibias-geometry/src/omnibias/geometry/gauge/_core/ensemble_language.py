# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Ensemble statistical language (Path B).

A finite-order jet of ``A`` and a per-configuration Wilson / Polyakov table
are the first two languages. Path B changes the object: rows are ensemble
statistics versus control parameters. One lattice configuration is not an
ensemble. GEVP / transfer-gap certificates are not search atoms.

Honesty: planted scaling laws or fixed-spacing lattice statistics. Not a
continuum string tension, not QCD, not a Yang-Mills mass gap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge.lattice._core.kernels import (
    polyakov_correlator_spatial,
    polyakov_loop_field,
)

ENSEMBLE_ABS_P = "abs_P"
ENSEMBLE_CHI_P = "chi_P"
ENSEMBLE_LOG_ABS_P = "log_abs_P"
ENSEMBLE_LOG_ABS_T = "log_abs_t"
ENSEMBLE_C_P = "C_P"
ENSEMBLE_LOG_C_P = "log_C_P"
ENSEMBLE_R = "r"
ENSEMBLE_G_P2 = "G_p2"
ENSEMBLE_P2 = "p2"
ENSEMBLE_RHO = "rho"
ENSEMBLE_OMEGA = "omega"

LEGAL_ENSEMBLE_ATOMS: frozenset[str] = frozenset(
    {
        ENSEMBLE_ABS_P,
        ENSEMBLE_CHI_P,
        ENSEMBLE_LOG_ABS_P,
        ENSEMBLE_LOG_ABS_T,
        ENSEMBLE_C_P,
        ENSEMBLE_LOG_C_P,
        ENSEMBLE_R,
        ENSEMBLE_G_P2,
        ENSEMBLE_P2,
        ENSEMBLE_RHO,
        ENSEMBLE_OMEGA,
    }
)

_CERT_ATOM_NAMES: frozenset[str] = frozenset(
    {
        "gevp",
        "gevp_plateau",
        "transfer_gap",
        "glueball_mass",
        "certified_transfer_matrix_gap",
    }
)

EnsembleSource = Literal["planted", "lattice_ensemble", "landau_gluon"]


def is_ensemble_atom_name(name: str) -> bool:
    """True for Path B statistical names (legal, cert, or obvious aliases)."""
    if name in LEGAL_ENSEMBLE_ATOMS or name in _CERT_ATOM_NAMES:
        return True
    key = name.lower().replace(" ", "")
    return key in {
        "abs_p",
        "chi_p",
        "log_abs_p",
        "log_abs_t",
        "c_p",
        "log_c_p",
        "g_p2",
        "g(p2)",
        "g(p^2)",
        "rho",
        "omega",
        "p2",
        "p^2",
    }


def is_ensemble_cert_name(name: str) -> bool:
    """True for GEVP / transfer-gap names that must not enter Path B STLSQ."""
    return name in _CERT_ATOM_NAMES or name.lower() in _CERT_ATOM_NAMES


def assert_library_ensemble_legal(
    names: Iterable[str],
    *,
    allow: frozenset[str] = LEGAL_ENSEMBLE_ATOMS,
) -> None:
    """Raise ``ValueError`` if any name is outside the closed ensemble allowlist."""
    illegal = [name for name in names if name not in allow]
    if illegal:
        raise ValueError(
            "ensemble library admits only allowlisted statistical atoms "
            f"{sorted(allow)}; rejected {illegal}"
        )


def refuse_jet_as_ensemble_source(
    obj: object | None = None, *_args: object, **_kwargs: object
) -> NoReturn:
    """A covariant / field jet is not an ensemble statistical observable."""
    _ = obj
    raise ValueError(
        "a gauge-covariant jet is not an ensemble source (Path B); "
        "statistical observables are not F/DF jets"
    )


def refuse_single_config_as_ensemble(
    obj: object | None = None, *_args: object, **_kwargs: object
) -> NoReturn:
    """One lattice configuration is not an ensemble."""
    _ = obj
    raise ValueError(
        "a single LatticeLinkField is not an ensemble (Path B); "
        "need two or more configurations, or a planted table"
    )


def refuse_loop_table_as_ensemble(
    obj: object | None = None, *_args: object, **_kwargs: object
) -> NoReturn:
    """Per-config Wilson / Polyakov traces are the holonomy language, not Path B."""
    _ = obj
    raise ValueError(
        "a LoopObservableTable is not an ensemble source (language split); "
        "Path B uses ensemble statistics, not per-config holonomy traces"
    )


def refuse_cert_as_ensemble_atom(
    names: Iterable[str] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """GEVP / transfer-gap certificates are not Path B search columns."""
    extra = f"; rejected {list(names)}" if names is not None else ""
    raise ValueError(
        "GEVP / transfer-gap certificates are not ensemble search atoms"
        f"{extra}"
    )


def refuse_green_as_ensemble_atom(
    names: Iterable[str] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """An inverse Laplacian is not an ensemble statistical atom."""
    extra = f"; rejected {list(names)}" if names is not None else ""
    raise ValueError(
        "an inverse Laplacian / Green function is not an ensemble atom"
        f"{extra}"
    )


def refuse_ensemble_as_covariant_jet(
    names: Iterable[str] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """Ensemble statistics are not a covariant jet."""
    extra = f"; rejected {list(names)}" if names is not None else ""
    raise ValueError(
        "ensemble statistical atoms are not a covariant jet (Path B); "
        "never mix abs_P / G_p2 / rho into F/DF singlets"
        f"{extra}"
    )


def refuse_ensemble_as_loop_source(
    names: Iterable[str] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """Ensemble statistics are not per-config holonomy traces."""
    extra = f"; rejected {list(names)}" if names is not None else ""
    raise ValueError(
        "ensemble statistical atoms are not loop traces (language split); "
        "never mix abs_P / G_p2 / rho into Wilson/Polyakov STLSQ"
        f"{extra}"
    )


@dataclass(frozen=True)
class EnsembleObservableTable:
    """Ensemble statistics. Not a jet and not a per-config loop table."""

    values: dict[str, np.ndarray]
    source: EnsembleSource = "planted"


def _as_link_array(field: LatticeLinkField) -> np.ndarray:
    links = np.asarray(field.links, dtype=np.float64)
    if links.ndim != 6 or links.shape[0] != 4 or links.shape[-1] != 4:
        raise ValueError(f"links must have shape (4, *L4, 4), got {links.shape}")
    if len(links.shape) - 2 != 4:
        raise ValueError(f"lattice must be 4-D, got shape {links.shape}")
    return links


def ensemble_table_from_link_ensemble(
    fields: Sequence[LatticeLinkField],
    *,
    beta: float,
    beta_c: float | None = None,
) -> EnsembleObservableTable:
    """Volume-mean ``|P|``, susceptibility, and ``C_P(r)`` from two or more configs."""
    if len(fields) < 2:
        refuse_single_config_as_ensemble(fields[0] if fields else None)
    p_means: list[float] = []
    corr_parts: list[np.ndarray] = []
    spatial_volume = 1.0
    for field in fields:
        links = _as_link_array(field)
        polyakov = np.asarray(polyakov_loop_field(np, links), dtype=np.float64)
        p_means.append(float(np.mean(polyakov)))
        corr_parts.append(
            np.asarray(polyakov_correlator_spatial(np, links), dtype=np.float64)
        )
        spatial_volume = float(np.prod(links.shape[1:4]))
    p_arr = np.asarray(p_means, dtype=np.float64)
    abs_p = float(np.mean(np.abs(p_arr)))
    chi_p = spatial_volume * float(np.mean(p_arr**2) - np.mean(p_arr) ** 2)
    corr = np.mean(np.stack(corr_parts, axis=0), axis=0)
    radii = np.arange(corr.shape[0], dtype=np.float64)
    values: dict[str, np.ndarray] = {
        ENSEMBLE_ABS_P: np.asarray([abs_p], dtype=np.float64),
        ENSEMBLE_CHI_P: np.asarray([chi_p], dtype=np.float64),
        ENSEMBLE_LOG_ABS_P: np.asarray(
            [float(np.log(max(abs_p, 1e-30)))], dtype=np.float64
        ),
        ENSEMBLE_C_P: corr,
        ENSEMBLE_LOG_C_P: np.log(np.maximum(np.abs(corr), 1e-30)),
        ENSEMBLE_R: radii,
    }
    _ = beta
    if beta_c is not None:
        reduced = abs((float(beta) - float(beta_c)) / float(beta_c))
        values[ENSEMBLE_LOG_ABS_T] = np.asarray(
            [float(np.log(max(reduced, 1e-30)))], dtype=np.float64
        )
    return EnsembleObservableTable(values=values, source="lattice_ensemble")


def ensemble_table_from_mc_dict(mc: Mapping[str, object]) -> EnsembleObservableTable:
    """Adapt a ``run_lattice_mc``-shaped dict. Does not run Monte Carlo."""
    values: dict[str, np.ndarray] = {}
    if "avg_polyakov" in mc:
        values[ENSEMBLE_ABS_P] = np.asarray(
            [abs(float(mc["avg_polyakov"]))], dtype=np.float64
        )
        values[ENSEMBLE_LOG_ABS_P] = np.log(np.maximum(values[ENSEMBLE_ABS_P], 1e-30))
    corr = mc.get("polyakov_correlator", mc.get("glueball_correlator"))
    if corr is not None:
        channel = np.asarray(corr, dtype=np.float64).reshape(-1)
        values[ENSEMBLE_C_P] = channel
        values[ENSEMBLE_R] = np.arange(channel.shape[0], dtype=np.float64)
        values[ENSEMBLE_LOG_C_P] = np.log(np.maximum(np.abs(channel), 1e-30))
    if "G_p2" in mc and "p2" in mc:
        values[ENSEMBLE_G_P2] = np.asarray(mc["G_p2"], dtype=np.float64).reshape(-1)
        values[ENSEMBLE_P2] = np.asarray(mc["p2"], dtype=np.float64).reshape(-1)
    if not values:
        raise ValueError(
            "mc dict has no ensemble atoms "
            "(avg_polyakov / correlator / G_p2)"
        )
    return EnsembleObservableTable(values=values, source="lattice_ensemble")


__all__ = [
    "ENSEMBLE_ABS_P",
    "ENSEMBLE_CHI_P",
    "ENSEMBLE_C_P",
    "ENSEMBLE_G_P2",
    "ENSEMBLE_LOG_ABS_P",
    "ENSEMBLE_LOG_ABS_T",
    "ENSEMBLE_LOG_C_P",
    "ENSEMBLE_OMEGA",
    "ENSEMBLE_P2",
    "ENSEMBLE_R",
    "ENSEMBLE_RHO",
    "EnsembleObservableTable",
    "EnsembleSource",
    "LEGAL_ENSEMBLE_ATOMS",
    "assert_library_ensemble_legal",
    "ensemble_table_from_link_ensemble",
    "ensemble_table_from_mc_dict",
    "is_ensemble_atom_name",
    "is_ensemble_cert_name",
    "refuse_cert_as_ensemble_atom",
    "refuse_ensemble_as_covariant_jet",
    "refuse_ensemble_as_loop_source",
    "refuse_green_as_ensemble_atom",
    "refuse_jet_as_ensemble_source",
    "refuse_loop_table_as_ensemble",
    "refuse_single_config_as_ensemble",
]
