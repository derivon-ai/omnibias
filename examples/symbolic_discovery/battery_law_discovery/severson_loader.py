# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Load Severson/TRI fast-charging battery data into a normalized table.

The original TRI release is stored as MATLAB `.mat` files. Public mirrors have
used both classic MATLAB and v7.3/HDF5 encodings, so the loader supports both
`scipy.io.loadmat` and `h5py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CycleTable:
    """Columnar battery cycle data used by the demo."""

    columns: dict[str, np.ndarray]

    def __len__(self) -> int:
        first = next(iter(self.columns.values()))
        return int(len(first))

    def require(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(f"cycle table has no column {name!r}; available={sorted(self.columns)}")
        return self.columns[name]

    def select_cells(self, cell_ids: set[str]) -> CycleTable:
        mask = np.isin(self.require("cell_id"), np.asarray(sorted(cell_ids), dtype=object))
        return CycleTable({k: v[mask] for k, v in self.columns.items()})


def find_mat_files(data_dir: Path) -> list[Path]:
    return sorted(Path(data_dir).expanduser().resolve().rglob("*.mat"))


def load_severson_cycle_table(data_dir: Path, max_cells: int | None = None) -> CycleTable:
    """Parse all `.mat` files below `data_dir`.

    If the real Severson structure cannot be extracted, a detailed error is
    raised instead of silently returning a malformed table.
    """
    mat_files = find_mat_files(data_dir)
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found under {data_dir}. Download with download_severson.py "
            "or pass --synthetic to run_demo.py."
        )

    cell_tables: list[dict[str, np.ndarray]] = []
    for mat_path in mat_files:
        extracted = _load_one_mat(mat_path)
        cell_tables.extend(extracted)
        if max_cells is not None and len(cell_tables) >= max_cells:
            cell_tables = cell_tables[:max_cells]
            break

    if not cell_tables:
        raise ValueError(f"Could not extract any battery cells from {data_dir}")
    return _concat_cell_tables(cell_tables)


def make_synthetic_cycle_table(
    n_cells: int = 24,
    n_cycles: int = 160,
    seed: int = 0,
) -> CycleTable:
    """Generate a small synthetic degradation dataset with known structure."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    for cell in range(n_cells):
        c_rate = rng.uniform(1.0, 6.0)
        temp = rng.uniform(24.0, 42.0)
        resistance = rng.uniform(0.015, 0.045)
        # Known degradation law used by the smoke test:
        # dq/dn = -(k0 + k_c*c_rate + k_t*temp + k_r*R) q
        k = 0.08 + 0.006 * c_rate + 0.0015 * (temp - 25.0) + 1.2 * resistance
        n = np.linspace(0.0, 1.0, n_cycles)
        q = np.exp(-k * n)
        q = q + rng.normal(0.0, 0.003, size=n_cycles)
        q = np.maximum.accumulate(q[::-1])[::-1]  # mostly monotone decline
        for i, (ni, qi) in enumerate(zip(n, q, strict=True), start=1):
            rows.append(
                {
                    "cell_id": f"synthetic_{cell:03d}",
                    "cycle_index": float(i),
                    "cycle_norm": float(ni),
                    "capacity_ah": float(2.0 * qi),
                    "capacity_norm": float(qi),
                    "c_rate": float(c_rate),
                    "temperature_c": float(temp),
                    "resistance_ohm": float(resistance),
                    "voltage_window": 1.2,
                }
            )
    return _rows_to_table(rows)


def train_test_cell_split(
    table: CycleTable,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[CycleTable, CycleTable]:
    cell_ids = np.unique(table.require("cell_id").astype(str))
    rng = np.random.default_rng(seed)
    shuffled = np.array(cell_ids)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    return table.select_cells(train_ids), table.select_cells(test_ids)


def train_test_protocol_split(
    table: CycleTable,
    test_fraction: float = 0.2,
) -> tuple[CycleTable, CycleTable]:
    """Hold out high-rate charging protocols for extrapolation testing."""
    c_rate = table.columns.get("c_rate")
    if c_rate is None or not np.isfinite(np.asarray(c_rate, dtype=float)).any():
        return train_test_cell_split(table, test_fraction=test_fraction, seed=0)

    cell_ids = np.unique(table.require("cell_id").astype(str))
    cell_rates = []
    for cell in cell_ids:
        mask = table.require("cell_id").astype(str) == cell
        rates = np.asarray(c_rate[mask], dtype=float)
        finite = rates[np.isfinite(rates)]
        rate = float(np.nanmedian(finite)) if finite.size else np.nan
        cell_rates.append((cell, rate))
    finite_pairs = [(cell, rate) for cell, rate in cell_rates if np.isfinite(rate)]
    if len(finite_pairs) < 4:
        return train_test_cell_split(table, test_fraction=test_fraction, seed=0)

    ordered = sorted(finite_pairs, key=lambda item: item[1])
    n_test = max(1, int(round(len(ordered) * test_fraction)))
    test_ids = {cell for cell, _ in ordered[-n_test:]}
    train_ids = set(cell_ids) - test_ids
    return table.select_cells(train_ids), table.select_cells(test_ids)


def _load_one_mat(mat_path: Path) -> list[dict[str, np.ndarray]]:
    try:
        return _load_scipy_mat(mat_path)
    except NotImplementedError:
        return _load_hdf5_mat(mat_path)
    except Exception as scipy_error:
        try:
            return _load_hdf5_mat(mat_path)
        except Exception as h5_error:
            raise ValueError(
                f"Failed to parse {mat_path} as scipy or HDF5 MATLAB data. "
                f"scipy error={scipy_error!r}; h5py error={h5_error!r}"
            ) from h5_error


def _load_scipy_mat(mat_path: Path) -> list[dict[str, np.ndarray]]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ValueError(
            "Parsing non-HDF5 .mat files requires scipy. Install `scipy` or use "
            "the HDF5 Severson mirror."
        ) from exc

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    batch = data.get("batch")
    if batch is None:
        raise ValueError(f"{mat_path} has no `batch` MATLAB struct")
    cells = np.ravel(batch)
    out = []
    for i, cell in enumerate(cells):
        out.append(_cell_from_mat_struct(cell, f"{mat_path.stem}_{i:03d}"))
    return out


def _cell_from_mat_struct(cell: Any, fallback_id: str) -> dict[str, np.ndarray]:
    summary = getattr(cell, "summary", None)
    cycle_life_raw = float(np.ravel(getattr(cell, "cycle_life", [0]))[0])
    policy = str(getattr(cell, "policy_readable", ""))

    q = _summary_vector(summary, ["QDischarge", "qDischarge", "capacity", "Capacity"])
    if q.size == 0:
        raise ValueError(f"cell {fallback_id} has no discharge capacity vector in summary")
    cycle_life = int(cycle_life_raw) if np.isfinite(cycle_life_raw) and cycle_life_raw > 1 else int(q.size)
    return _table_for_cell(
        cell_id=fallback_id,
        q_discharge=q,
        cycle_life=cycle_life,
        policy=policy,
        temperature=_summary_vector(summary, ["Tavg", "Tmean", "Temperature_measured"]),
        resistance=_summary_vector(summary, ["IR", "internal_resistance", "Resistance"]),
        charge_capacity=_summary_vector(summary, ["QCharge", "qCharge"]),
    )


def _summary_vector(summary: Any, names: list[str]) -> np.ndarray:
    if summary is None:
        return np.asarray([], dtype=float)
    for name in names:
        if hasattr(summary, name):
            arr = np.asarray(getattr(summary, name), dtype=float).reshape(-1)
            if arr.size:
                return arr
        if isinstance(summary, dict) and name in summary:
            arr = np.asarray(summary[name], dtype=float).reshape(-1)
            if arr.size:
                return arr
    return np.asarray([], dtype=float)


def _load_hdf5_mat(mat_path: Path) -> list[dict[str, np.ndarray]]:
    try:
        import h5py
    except ImportError as exc:
        raise ValueError("Parsing MATLAB v7.3/HDF5 files requires h5py.") from exc

    with h5py.File(mat_path, "r") as h5:
        if "batch" not in h5:
            raise ValueError(f"{mat_path} has no HDF5 `batch` group")
        batch = h5["batch"]
        if "summary" not in batch:
            raise ValueError(f"{mat_path} batch has no `summary` references")
        n_cells = _ref_column(batch["summary"]).shape[0]
        out = []
        for i in range(n_cells):
            summary_group = h5[_decode_ref(batch["summary"], i)]
            q = _h5_summary_vector(h5, summary_group, ["QDischarge", "qDischarge", "capacity"])
            if q.size == 0:
                continue
            cycle_life = _h5_scalar_ref(h5, batch, "cycle_life", i, default=len(q))
            if not np.isfinite(cycle_life) or cycle_life <= 1:
                cycle_life = float(len(q))
            policy = _h5_string_ref(h5, batch, "policy_readable", i, default="")
            out.append(
                _table_for_cell(
                    cell_id=f"{mat_path.stem}_{i:03d}",
                    q_discharge=q,
                    cycle_life=int(cycle_life),
                    policy=policy,
                    temperature=_h5_summary_vector(h5, summary_group, ["Tavg", "Tmean"]),
                    resistance=_h5_summary_vector(h5, summary_group, ["IR", "internal_resistance"]),
                    charge_capacity=_h5_summary_vector(h5, summary_group, ["QCharge", "qCharge"]),
                )
            )
        return out


def _ref_column(dataset: Any) -> np.ndarray:
    arr = np.asarray(dataset)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0, :]
    return arr.reshape(-1)


def _decode_ref(dataset: Any, i: int) -> Any:
    return _ref_column(dataset)[i]


def _h5_scalar_ref(h5: Any, batch: Any, name: str, i: int, default: float) -> float:
    if name not in batch:
        return float(default)
    ref = _decode_ref(batch[name], i)
    arr = np.asarray(h5[ref]).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _h5_string_ref(h5: Any, batch: Any, name: str, i: int, default: str) -> str:
    if name not in batch:
        return default
    ref = _decode_ref(batch[name], i)
    raw = np.asarray(h5[ref]).reshape(-1)
    try:
        return "".join(chr(int(x)) for x in raw if int(x) != 0)
    except Exception:
        return default


def _h5_summary_vector(h5: Any, summary_group: Any, names: list[str]) -> np.ndarray:
    for name in names:
        if name not in summary_group:
            continue
        dataset = summary_group[name]
        raw = np.asarray(dataset)
        if raw.dtype != object:
            values = raw.reshape(-1).astype(float)
            if values.size:
                return values
        refs = raw.reshape(-1)
        values: list[float] = []
        for ref in refs:
            arr = np.asarray(h5[ref]).reshape(-1)
            if arr.size:
                values.append(float(arr[0]))
        if values:
            return np.asarray(values, dtype=float)
    return np.asarray([], dtype=float)


def _table_for_cell(
    cell_id: str,
    q_discharge: np.ndarray,
    cycle_life: int,
    policy: str,
    temperature: np.ndarray,
    resistance: np.ndarray,
    charge_capacity: np.ndarray,
) -> dict[str, np.ndarray]:
    q_raw = np.asarray(q_discharge, dtype=float).reshape(-1)
    valid = np.isfinite(q_raw) & (q_raw > 0.05)
    if valid.any():
        q = q_raw[valid]
        temperature = _filter_cycle_vector(temperature, valid)
        resistance = _filter_cycle_vector(resistance, valid)
        charge_capacity = _filter_cycle_vector(charge_capacity, valid)
    else:
        q = q_raw
    n = q.size
    cycle = np.arange(1, n + 1, dtype=float)
    q0 = float(np.nanmedian(q[: min(5, n)]))
    q_norm = q / max(q0, 1e-12)
    temp = _broadcast_or_nan(temperature, n, fallback=np.nan)
    res = _broadcast_or_nan(resistance, n, fallback=np.nan)
    q_charge = _broadcast_or_nan(charge_capacity, n, fallback=np.nan)
    c_rate, voltage_window = _policy_features(policy)
    return {
        "cell_id": np.full(n, cell_id, dtype=object),
        "cycle_index": cycle,
        "cycle_norm": (cycle - 1.0) / max(float(cycle_life - 1), 1.0),
        "capacity_ah": q,
        "capacity_norm": q_norm,
        "temperature_c": temp,
        "resistance_ohm": res,
        "charge_capacity_ah": q_charge,
        "c_rate": np.full(n, c_rate),
        "voltage_window": np.full(n, voltage_window),
    }


def _broadcast_or_nan(values: np.ndarray, n: int, fallback: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    return np.full(n, fallback)


def _filter_cycle_vector(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == valid.size:
        return arr[valid]
    return arr


def _policy_features(policy: str) -> tuple[float, float]:
    # Severson policy strings encode rates such as "4.8C(80%)-4.8C".
    import re

    rates = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)C", policy)]
    c_rate = float(np.mean(rates)) if rates else np.nan
    voltage_window = 1.2  # LFP protocol default when no explicit voltage range is available.
    return c_rate, voltage_window


def _concat_cell_tables(cell_tables: list[dict[str, np.ndarray]]) -> CycleTable:
    keys = sorted(set().union(*(table.keys() for table in cell_tables)))
    cols = {}
    for key in keys:
        arrays = []
        for table in cell_tables:
            if key in table:
                arrays.append(table[key])
            else:
                arrays.append(np.full(len(table["cycle_index"]), np.nan))
        cols[key] = np.concatenate(arrays, axis=0)
    return CycleTable(cols)


def _rows_to_table(rows: list[dict[str, float | str]]) -> CycleTable:
    keys = sorted(rows[0])
    cols: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row[key] for row in rows]
        dtype = object if isinstance(values[0], str) else float
        cols[key] = np.asarray(values, dtype=dtype)
    return CycleTable(cols)
