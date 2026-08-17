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
from dataclasses import dataclass, field
from typing import Literal, NoReturn

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge.lattice._core.kernels import (
    polyakov_correlator_spatial,
    polyakov_loop_field,
)
from omnibias.geometry.gauge.lattice._core.stats import creutz_ratio_value

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
ENSEMBLE_AREA = "area"
ENSEMBLE_PERIMETER = "perimeter"
ENSEMBLE_LOG_P2 = "log_p2"
ENSEMBLE_INV_P2 = "inv_p2"
ENSEMBLE_INV_P2_SQ = "inv_p2_sq"
ENSEMBLE_GHOST_G = "ghost_G"
ENSEMBLE_LOG_GHOST_G = "log_ghost_G"
ENSEMBLE_T_LAT = "T_lat"
ENSEMBLE_V_R = "V_r"
ENSEMBLE_T_WILSON = "t_wilson"
ENSEMBLE_CREUTZ_CHI = "creutz_chi"
ENSEMBLE_L_LAT = "L_lat"
ENSEMBLE_F_R = "F_r"
ENSEMBLE_SIGMA_LAT = "sigma_lat"
ENSEMBLE_LAMBDA_QCD = "Lambda_QCD"
ENSEMBLE_R0 = "r0"
ENSEMBLE_T0 = "t0"
ENSEMBLE_A_LAT = "a_lat"
ENSEMBLE_LOG_P2_OVER_L2 = "log_p2_over_L2"
ENSEMBLE_P2_G025 = "p2_g025"
ENSEMBLE_P2_G05 = "p2_g05"
ENSEMBLE_P2_G075 = "p2_g075"
ENSEMBLE_P2_G1 = "p2_g1"
ENSEMBLE_LOG_P2_G025 = "log_p2_g025"
ENSEMBLE_LOG_P2_G05 = "log_p2_g05"
ENSEMBLE_LOG_P2_G075 = "log_p2_g075"
ENSEMBLE_LOG_P2_G1 = "log_p2_g1"
ENSEMBLE_LI2_Z = "Li2_z"
ENSEMBLE_LI3_Z = "Li3_z"
ENSEMBLE_F21_Z = "F21_z"
ENSEMBLE_GLUEBALL_MASS = "glueball_mass"

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
        ENSEMBLE_AREA,
        ENSEMBLE_PERIMETER,
        ENSEMBLE_LOG_P2,
        ENSEMBLE_INV_P2,
        ENSEMBLE_INV_P2_SQ,
        ENSEMBLE_GHOST_G,
        ENSEMBLE_LOG_GHOST_G,
        ENSEMBLE_T_LAT,
        ENSEMBLE_V_R,
        ENSEMBLE_T_WILSON,
        ENSEMBLE_CREUTZ_CHI,
        ENSEMBLE_L_LAT,
        ENSEMBLE_F_R,
        ENSEMBLE_SIGMA_LAT,
        ENSEMBLE_LAMBDA_QCD,
        ENSEMBLE_R0,
        ENSEMBLE_T0,
        ENSEMBLE_A_LAT,
        ENSEMBLE_LOG_P2_OVER_L2,
        ENSEMBLE_P2_G025,
        ENSEMBLE_P2_G05,
        ENSEMBLE_P2_G075,
        ENSEMBLE_P2_G1,
        ENSEMBLE_LOG_P2_G025,
        ENSEMBLE_LOG_P2_G05,
        ENSEMBLE_LOG_P2_G075,
        ENSEMBLE_LOG_P2_G1,
        ENSEMBLE_LI2_Z,
        ENSEMBLE_LI3_Z,
        ENSEMBLE_F21_Z,
        ENSEMBLE_GLUEBALL_MASS,
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


@dataclass(frozen=True)
class LatticeMetadata:
    """Lattice bookkeeping on a Path B table. Not a continuum claim."""

    beta: float | None = None
    spacing: float | None = None
    lattice_shape: tuple[int, ...] | None = None
    gauge_group: str = "su(2)"
    scheme: str = ""
    already_inverse: bool = False
    n_configs: int = 0
    clover: float | None = None


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
        "area",
        "perimeter",
        "log_p2",
        "inv_p2",
        "inv_p2_sq",
        "ghost_g",
        "log_ghost_g",
        "t_lat",
        "v_r",
        "t_wilson",
        "creutz_chi",
        "l_lat",
        "f_r",
        "sigma_lat",
        "lambda_qcd",
        "r0",
        "t0",
        "a_lat",
        "log_p2_over_l2",
        "p2_g025",
        "p2_g05",
        "p2_g075",
        "p2_g1",
        "log_p2_g025",
        "log_p2_g05",
        "log_p2_g075",
        "log_p2_g1",
        "li2_z",
        "li3_z",
        "f21_z",
        "glueball_mass",
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
    metadata: LatticeMetadata = field(default_factory=LatticeMetadata)


def _as_link_array(field: LatticeLinkField) -> np.ndarray:
    links = np.asarray(field.links, dtype=np.float64)
    if links.ndim != 6 or links.shape[0] != 4 or links.shape[-1] != 4:
        raise ValueError(f"links must have shape (4, *L4, 4), got {links.shape}")
    if len(links.shape) - 2 != 4:
        raise ValueError(f"lattice must be 4-D, got shape {links.shape}")
    return links


def _finite_t_lat(shape: tuple[int, ...]) -> float | None:
    if len(shape) != 4:
        return None
    n_s = min(int(shape[0]), int(shape[1]), int(shape[2]))
    n_t = int(shape[3])
    if n_t < n_s:
        return 1.0 / float(n_t)
    return None


def spatial_l_lat(shape: tuple[int, ...] | None) -> float | None:
    """Minimum spatial extent. Not a physical volume."""
    if shape is None or len(shape) < 3:
        return None
    return float(min(int(shape[0]), int(shape[1]), int(shape[2])))


def _broadcast(value: float, n_rows: int) -> np.ndarray:
    return np.full(int(n_rows), float(value), dtype=np.float64)


def _attach_l_lat(
    values: dict[str, np.ndarray],
    lattice_shape: tuple[int, ...] | None,
    *,
    n_rows: int | None = None,
) -> None:
    length = spatial_l_lat(lattice_shape)
    if length is None:
        return
    if n_rows is None:
        n_rows = max(int(np.asarray(arr).reshape(-1).shape[0]) for arr in values.values())
    values[ENSEMBLE_L_LAT] = _broadcast(length, n_rows)


def _parse_rx_t(key: object) -> tuple[int, int]:
    if isinstance(key, tuple) and len(key) == 2:
        return int(key[0]), int(key[1])
    text = str(key).lower().replace("×", "x")
    if "x" not in text:
        raise ValueError(f"Wilson key must look like 'RxT', got {key!r}")
    r_s, t_s = text.split("x", 1)
    return int(r_s), int(t_s)


def _wilson_mean(payload: object) -> float:
    if isinstance(payload, Mapping):
        if "value" in payload:
            return float(payload["value"])
        raise ValueError("wilson payload mapping needs a 'value' key")
    arr = np.asarray(payload, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("wilson payload is empty")
    return float(np.mean(arr))


def _centered_force(radii: np.ndarray, potential: np.ndarray) -> np.ndarray:
    """Three-point ``dV/dr`` on unique sorted radii, mapped back to rows."""
    order = np.argsort(radii)
    r_sorted = radii[order]
    v_sorted = potential[order]
    force = np.zeros_like(v_sorted)
    n = int(r_sorted.shape[0])
    if n == 1:
        return force
    force[0] = (v_sorted[1] - v_sorted[0]) / max(r_sorted[1] - r_sorted[0], 1e-30)
    force[-1] = (v_sorted[-1] - v_sorted[-2]) / max(r_sorted[-1] - r_sorted[-2], 1e-30)
    if n > 2:
        force[1:-1] = (v_sorted[2:] - v_sorted[:-2]) / np.maximum(
            r_sorted[2:] - r_sorted[:-2], 1e-30
        )
    out = np.empty_like(potential)
    out[order] = force
    return out


def wilson_loops_to_ensemble_table(
    wilson_loops: Mapping[object, object],
    *,
    lattice_shape: tuple[int, ...] | None = None,
    beta: float | None = None,
    spacing: float | None = None,
    gauge_group: str = "su(2)",
    source: EnsembleSource = "lattice_ensemble",
    n_configs: int = 0,
) -> EnsembleObservableTable:
    """Turn ``RxT`` Wilson means into a Path B table.

    Accepts ``wilson_loops_ensemble`` output (``\"RxT\" → {value, err}``),
    raw sample lists, or scalar means. Adds Creutz ``χ(R,T)`` where the
    four neighbouring rectangles exist. Not a continuum string tension.
    """
    if not wilson_loops:
        raise ValueError("wilson_loops is empty")
    rows: list[tuple[int, int, float]] = []
    for key, payload in wilson_loops.items():
        r_val, t_val = _parse_rx_t(key)
        rows.append((r_val, t_val, _wilson_mean(payload)))
    rows.sort()
    radii = np.asarray([row[0] for row in rows], dtype=np.float64)
    times = np.asarray([row[1] for row in rows], dtype=np.float64)
    weights = np.asarray([row[2] for row in rows], dtype=np.float64)
    lookup = {(int(row[0]), int(row[1])): float(row[2]) for row in rows}
    creutz = np.full(radii.shape, np.nan, dtype=np.float64)
    for i, (r_val, t_val, _) in enumerate(rows):
        if r_val < 2 or t_val < 2:
            continue
        needed = (
            (r_val, t_val),
            (r_val - 1, t_val - 1),
            (r_val - 1, t_val),
            (r_val, t_val - 1),
        )
        if any(item not in lookup for item in needed):
            continue
        creutz[i] = creutz_ratio_value(
            lookup[(r_val, t_val)],
            lookup[(r_val - 1, t_val - 1)],
            lookup[(r_val - 1, t_val)],
            lookup[(r_val, t_val - 1)],
        )
    finite_chi = creutz[np.isfinite(creutz)]
    sigma_lat = float(np.mean(finite_chi)) if finite_chi.size else float("nan")
    values: dict[str, np.ndarray] = {
        ENSEMBLE_R: radii,
        ENSEMBLE_T_WILSON: times,
        ENSEMBLE_AREA: radii * times,
        ENSEMBLE_PERIMETER: 2.0 * (radii + times),
        ENSEMBLE_C_P: weights,
        ENSEMBLE_LOG_C_P: np.log(np.maximum(np.abs(weights), 1e-30)),
        ENSEMBLE_CREUTZ_CHI: creutz,
        ENSEMBLE_SIGMA_LAT: _broadcast(sigma_lat, radii.shape[0]),
    }
    _attach_l_lat(values, lattice_shape, n_rows=int(radii.shape[0]))
    table = EnsembleObservableTable(
        values=values,
        source=source,
        metadata=LatticeMetadata(
            beta=beta,
            spacing=spacing,
            lattice_shape=lattice_shape,
            gauge_group=gauge_group,
            scheme="wilson",
            n_configs=n_configs,
        ),
    )
    return static_potential_from_wilson(table)


def static_potential_from_wilson(
    table: EnsembleObservableTable,
) -> EnsembleObservableTable:
    """``V(r,T) = -log W / T``; ``V(r)`` is the large-``T`` plateau.

    Writes ``V_r`` and a centered ``F_r = dV/dr``. Lattice units only.
    """
    values = {name: np.asarray(arr, dtype=float).reshape(-1) for name, arr in table.values.items()}
    if ENSEMBLE_LOG_C_P not in values or ENSEMBLE_T_WILSON not in values:
        raise ValueError("static potential needs log_C_P and t_wilson")
    if ENSEMBLE_R not in values:
        raise ValueError("static potential needs r")
    log_w = values[ENSEMBLE_LOG_C_P]
    times = np.maximum(values[ENSEMBLE_T_WILSON], 1e-30)
    radii = values[ENSEMBLE_R]
    v_rt = -log_w / times
    plateau = np.empty_like(v_rt)
    unique_r = np.unique(radii)
    for radius in unique_r:
        mask = radii == radius
        t_vals = times[mask]
        v_vals = v_rt[mask]
        order = np.argsort(t_vals)
        upper = max(int(np.ceil(order.shape[0] / 2.0)), 1)
        plateau[mask] = float(np.median(v_vals[order[-upper:]]))
    values[ENSEMBLE_V_R] = plateau
    unique_mask = np.zeros(radii.shape[0], dtype=bool)
    seen: set[float] = set()
    for i, radius in enumerate(radii):
        key = float(radius)
        if key not in seen:
            unique_mask[i] = True
            seen.add(key)
    force_unique = _centered_force(radii[unique_mask], plateau[unique_mask])
    force = np.empty_like(plateau)
    lookup = {
        float(r): float(f)
        for r, f in zip(radii[unique_mask], force_unique, strict=True)
    }
    for i, radius in enumerate(radii):
        force[i] = lookup[float(radius)]
    values[ENSEMBLE_F_R] = force
    return EnsembleObservableTable(
        values=values,
        source=table.source,
        metadata=table.metadata,
    )


def ensemble_table_from_link_ensemble(
    fields: Sequence[LatticeLinkField],
    *,
    beta: float,
    beta_c: float | None = None,
    spacing: float | None = None,
    gauge_group: str = "su(2)",
) -> EnsembleObservableTable:
    """Volume-mean ``|P|``, susceptibility, and ``C_P(r)`` from two or more configs."""
    if len(fields) < 2:
        refuse_single_config_as_ensemble(fields[0] if fields else None)
    p_means: list[float] = []
    corr_parts: list[np.ndarray] = []
    spatial_volume = 1.0
    lattice_shape: tuple[int, ...] | None = None
    for field_item in fields:
        links = _as_link_array(field_item)
        lattice_shape = tuple(int(size) for size in links.shape[1:5])
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
    t_lat = _finite_t_lat(lattice_shape) if lattice_shape is not None else None
    if t_lat is not None:
        values[ENSEMBLE_T_LAT] = np.asarray([t_lat], dtype=np.float64)
    _attach_l_lat(values, lattice_shape, n_rows=1)
    if beta_c is not None:
        reduced = abs((float(beta) - float(beta_c)) / float(beta_c))
        values[ENSEMBLE_LOG_ABS_T] = np.asarray(
            [float(np.log(max(reduced, 1e-30)))], dtype=np.float64
        )
    return EnsembleObservableTable(
        values=values,
        source="lattice_ensemble",
        metadata=LatticeMetadata(
            beta=float(beta),
            spacing=spacing,
            lattice_shape=lattice_shape,
            gauge_group=gauge_group,
            scheme="wilson",
            n_configs=len(fields),
        ),
    )


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
        p2 = np.asarray(mc["p2"], dtype=np.float64).reshape(-1)
        values[ENSEMBLE_G_P2] = np.asarray(mc["G_p2"], dtype=np.float64).reshape(-1)
        values[ENSEMBLE_P2] = p2
        values[ENSEMBLE_LOG_P2] = np.log(np.maximum(p2, 1e-30))
        values[ENSEMBLE_INV_P2] = 1.0 / np.maximum(p2, 1e-30)
    if "ghost_G" in mc and "p2" in mc:
        ghost = np.asarray(mc["ghost_G"], dtype=np.float64).reshape(-1)
        values[ENSEMBLE_GHOST_G] = ghost
        values[ENSEMBLE_LOG_GHOST_G] = np.log(np.maximum(np.abs(ghost), 1e-30))
        if ENSEMBLE_P2 not in values:
            p2 = np.asarray(mc["p2"], dtype=np.float64).reshape(-1)
            values[ENSEMBLE_P2] = p2
    wilson_loops = mc.get("wilson_loops")
    shape_raw = mc.get("lattice_shape")
    lattice_shape = None
    if isinstance(shape_raw, (list, tuple)):
        lattice_shape = tuple(int(size) for size in shape_raw)
        t_lat = _finite_t_lat(lattice_shape)
        if t_lat is not None:
            values[ENSEMBLE_T_LAT] = np.asarray([t_lat], dtype=np.float64)
    if isinstance(wilson_loops, Mapping) and wilson_loops:
        wilson_table = wilson_loops_to_ensemble_table(
            wilson_loops,
            lattice_shape=lattice_shape,
            beta=None if mc.get("beta") is None else float(mc["beta"]),
            gauge_group=str(mc.get("gauge_group") or "su(2)"),
            n_configs=int(mc.get("n_meas") or 0),
        )
        for name, arr in wilson_table.values.items():
            values.setdefault(name, arr)
        if not values:
            return wilson_table
    _attach_l_lat(values, lattice_shape)
    if not values:
        raise ValueError(
            "mc dict has no ensemble atoms "
            "(avg_polyakov / correlator / G_p2 / wilson_loops)"
        )
    beta = mc.get("beta")
    return EnsembleObservableTable(
        values=values,
        source="lattice_ensemble",
        metadata=LatticeMetadata(
            beta=None if beta is None else float(beta),
            lattice_shape=lattice_shape,
            gauge_group=str(mc.get("gauge_group") or "su(2)"),
            scheme="wilson",
            n_configs=int(mc.get("n_meas") or 0),
        ),
    )


def finite_t_scan_table(
    *,
    t_c: float = 1.0,
    n_rows: int = 16,
    sigma0: float = 0.2,
    p0: float = 0.4,
) -> EnsembleObservableTable:
    """Planted ``|P|(T)`` / ``σ(T)`` scan for the joint discoverer.

    Below ``T_c``, ``σ = σ0 (1 - T/T_c)`` and ``|P|`` is small; above, ``σ=0``
    and ``|P|`` melts on. Not a continuum critical temperature.
    """
    temps = np.linspace(0.4 * t_c, 1.4 * t_c, n_rows, dtype=np.float64)
    reduced = np.clip(1.0 - temps / t_c, 0.0, None)
    sigma = sigma0 * reduced
    above = np.sqrt(np.maximum(temps / t_c - 1.0, 0.0))
    abs_p = np.where(temps < t_c, 1e-3 + 0.05 * (temps / t_c), p0 * np.maximum(above, 1e-3))
    abs_p = np.clip(abs_p, 1e-6, None)
    return EnsembleObservableTable(
        values={
            ENSEMBLE_T_LAT: temps,
            ENSEMBLE_ABS_P: abs_p,
            ENSEMBLE_LOG_ABS_P: np.log(abs_p),
            ENSEMBLE_SIGMA_LAT: sigma,
        },
        source="planted",
        metadata=LatticeMetadata(scheme="finite_t_scan", n_configs=n_rows),
    )


__all__ = [
    "ENSEMBLE_ABS_P",
    "ENSEMBLE_AREA",
    "ENSEMBLE_CHI_P",
    "ENSEMBLE_C_P",
    "ENSEMBLE_A_LAT",
    "ENSEMBLE_CREUTZ_CHI",
    "ENSEMBLE_F21_Z",
    "ENSEMBLE_F_R",
    "ENSEMBLE_GHOST_G",
    "ENSEMBLE_GLUEBALL_MASS",
    "ENSEMBLE_G_P2",
    "ENSEMBLE_INV_P2",
    "ENSEMBLE_INV_P2_SQ",
    "ENSEMBLE_LAMBDA_QCD",
    "ENSEMBLE_LI2_Z",
    "ENSEMBLE_LI3_Z",
    "ENSEMBLE_L_LAT",
    "ENSEMBLE_LOG_ABS_P",
    "ENSEMBLE_LOG_ABS_T",
    "ENSEMBLE_LOG_C_P",
    "ENSEMBLE_LOG_GHOST_G",
    "ENSEMBLE_LOG_P2",
    "ENSEMBLE_LOG_P2_G025",
    "ENSEMBLE_LOG_P2_G05",
    "ENSEMBLE_LOG_P2_G075",
    "ENSEMBLE_LOG_P2_G1",
    "ENSEMBLE_LOG_P2_OVER_L2",
    "ENSEMBLE_OMEGA",
    "ENSEMBLE_P2",
    "ENSEMBLE_P2_G025",
    "ENSEMBLE_P2_G05",
    "ENSEMBLE_P2_G075",
    "ENSEMBLE_P2_G1",
    "ENSEMBLE_PERIMETER",
    "ENSEMBLE_R",
    "ENSEMBLE_R0",
    "ENSEMBLE_RHO",
    "ENSEMBLE_SIGMA_LAT",
    "ENSEMBLE_T0",
    "ENSEMBLE_T_LAT",
    "ENSEMBLE_T_WILSON",
    "ENSEMBLE_V_R",
    "EnsembleObservableTable",
    "EnsembleSource",
    "LEGAL_ENSEMBLE_ATOMS",
    "LatticeMetadata",
    "assert_library_ensemble_legal",
    "ensemble_table_from_link_ensemble",
    "ensemble_table_from_mc_dict",
    "finite_t_scan_table",
    "is_ensemble_atom_name",
    "is_ensemble_cert_name",
    "refuse_cert_as_ensemble_atom",
    "refuse_ensemble_as_covariant_jet",
    "refuse_ensemble_as_loop_source",
    "refuse_green_as_ensemble_atom",
    "refuse_jet_as_ensemble_source",
    "refuse_loop_table_as_ensemble",
    "refuse_single_config_as_ensemble",
    "spatial_l_lat",
    "static_potential_from_wilson",
    "wilson_loops_to_ensemble_table",
]
