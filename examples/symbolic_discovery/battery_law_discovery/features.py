# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Feature engineering for the battery law-discovery demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # script execution fallback
    from .severson_loader import CycleTable
except ImportError:  # pragma: no cover
    from severson_loader import CycleTable


@dataclass(frozen=True)
class FeatureBundle:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    cell_id: np.ndarray
    cycle_norm: np.ndarray
    capacity_norm: np.ndarray


@dataclass(frozen=True)
class FeatureStats:
    fill_values: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]


def fit_feature_stats(table: CycleTable) -> FeatureStats:
    """Fit feature imputation and scaling stats on the training table only."""
    raw = _raw_feature_columns(table)
    fill_values: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name, default in [
        ("temperature_c", 30.0),
        ("resistance_ohm", 0.03),
        ("c_rate", 3.0),
        ("voltage_window", 1.2),
    ]:
        values = np.asarray(table.columns.get(name, [default]), dtype=float).reshape(-1)
        finite = np.isfinite(values)
        fill_values[name] = float(np.nanmedian(values[finite])) if finite.any() else default
    filled = {
        name: _fill_with_value(table.columns.get(name), fill_values[name], len(table))
        for name in fill_values
    }
    raw.update(
        {
            "c_rate_z": filled["c_rate"],
            "temperature_z": filled["temperature_c"],
            "resistance_z": filled["resistance_ohm"],
            "voltage_window_z": filled["voltage_window"],
        }
    )
    for name in ["c_rate_z", "temperature_z", "resistance_z", "voltage_window_z"]:
        arr = raw[name]
        means[name] = float(np.mean(arr))
        scale = float(np.std(arr))
        scales[name] = scale if scale >= 1e-12 else 1.0
    return FeatureStats(fill_values=fill_values, means=means, scales=scales)


def build_feature_bundle(table: CycleTable, stats: FeatureStats | None = None) -> FeatureBundle:
    """Build numeric features for the omnibias field and baselines."""
    if stats is None:
        stats = fit_feature_stats(table)
    cycle_norm = table.require("cycle_norm").astype(float)
    q = table.require("capacity_norm").astype(float)

    temp = _fill_with_value(table.columns.get("temperature_c"), stats.fill_values["temperature_c"], len(table))
    resistance = _fill_with_value(table.columns.get("resistance_ohm"), stats.fill_values["resistance_ohm"], len(table))
    c_rate = _fill_with_value(table.columns.get("c_rate"), stats.fill_values["c_rate"], len(table))
    voltage_window = _fill_with_value(
        table.columns.get("voltage_window"), stats.fill_values["voltage_window"], len(table)
    )

    # Normalize non-cycle inputs to make the one-layer field well-conditioned.
    raw = {
        "cycle_norm": cycle_norm,
        "c_rate_z": _scale_with_stats(c_rate, stats, "c_rate_z"),
        "temperature_z": _scale_with_stats(temp, stats, "temperature_z"),
        "resistance_z": _scale_with_stats(resistance, stats, "resistance_z"),
        "voltage_window_z": _scale_with_stats(voltage_window, stats, "voltage_window_z"),
    }
    names = list(raw)
    x = np.stack([raw[name] for name in names], axis=-1).astype(np.float64)
    return FeatureBundle(
        x=x,
        y=q.astype(np.float64),
        feature_names=names,
        cell_id=table.require("cell_id").astype(str),
        cycle_norm=cycle_norm.astype(np.float64),
        capacity_norm=q.astype(np.float64),
    )


def early_cycle_features(bundle: FeatureBundle, max_cycle_norm: float = 0.2) -> dict[str, dict[str, float]]:
    """Summarize early-cycle behavior per cell for reporting and baselines."""
    out: dict[str, dict[str, float]] = {}
    for cell in np.unique(bundle.cell_id):
        mask = (bundle.cell_id == cell) & (bundle.cycle_norm <= max_cycle_norm)
        if mask.sum() < 3:
            mask = bundle.cell_id == cell
        n = bundle.cycle_norm[mask]
        q = bundle.capacity_norm[mask]
        slope = float(np.polyfit(n, q, 1)[0]) if q.size >= 2 else 0.0
        out[cell] = {
            "q0": float(q[0]),
            "q_last_early": float(q[-1]),
            "early_slope": slope,
            "early_std": float(np.std(q)),
        }
    return out


def cell_indices(cell_id: np.ndarray) -> dict[str, np.ndarray]:
    return {cell: np.flatnonzero(cell_id == cell) for cell in np.unique(cell_id)}


def _raw_feature_columns(table: CycleTable) -> dict[str, np.ndarray]:
    return {"cycle_norm": table.require("cycle_norm").astype(float)}


def _fill_with_value(values: np.ndarray | None, fill: float, n: int) -> np.ndarray:
    if values is None:
        return np.full(n, fill, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 1 and n != 1:
        arr = np.full(n, float(arr[0]))
    finite = np.isfinite(arr)
    return np.where(finite, arr, fill)


def _scale_with_stats(values: np.ndarray, stats: FeatureStats, name: str) -> np.ndarray:
    return (np.asarray(values, dtype=float) - stats.means[name]) / stats.scales[name]
