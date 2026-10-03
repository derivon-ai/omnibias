#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Extract the five coefficient-independent k=39 interval kernels.

The extractor imports one generated PrimeGaps evaluator, checks that its
companion receipt binds the pinned source digest, and builds the denominator,
J0, Jplus, Jtail, and source-loss forms as 77-by-77 symmetric interval
matrices.  A positive later screen is not a crossing.  This script does not
copy or fit a k=40 quadratic form.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

PINNED_SOURCE_SHA256 = (
    "7f71bdefcfe3bb5ca76a143929b3cb3f4156c21dc483253cda3077420f1e5de4"
)
VARIABLES = 77
UPPER_COUNT = VARIABLES * (VARIABLES + 1) // 2

_PG = None
_ENGINE = None
_BANK = None
_BASIS_INDEX = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("prime_gap_k39_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_evaluator(path: Path) -> tuple[object, str]:
    """Refuse a generator output that is not tied to the pinned digest."""

    receipt_path = path.with_suffix(".receipt.json")
    if not receipt_path.is_file():
        raise RuntimeError(
            "generated evaluator has no rewrite receipt; "
            "refusing unbound quadratic kernels"
        )
    meta = json.loads(receipt_path.read_text(encoding="utf-8"))
    if meta.get("upstream_source_sha256") != PINNED_SOURCE_SHA256:
        raise RuntimeError("evaluator receipt is not bound to the pinned source digest")
    if meta.get("upstream_commit") != "61340d0b74163003b32756bb16e91d9209a5e330":
        raise RuntimeError("evaluator receipt is not bound to the pinned commit")
    transformed = hashlib.sha256(path.read_bytes()).hexdigest()
    if transformed != meta.get("transformed_source_sha256"):
        raise RuntimeError("generated evaluator bytes do not match its rewrite receipt")
    module = _load_module(path)
    if int(module.TRIAL["dimension"]) != 39:
        raise RuntimeError("quadratic extraction requires the dimension-39 evaluator")
    if module.POLICY["convolution_length"] != 98265:
        raise RuntimeError("dimension-39 convolution length changed")
    module.check_flint_signed_fft()
    return module, transformed


def _upper_index(row: int, column: int) -> int:
    if row > column:
        row, column = column, row
    return row * VARIABLES - row * (row - 1) // 2 + (column - row)


def _zero_upper(pg) -> list:
    return [pg.arb(0) for _ in range(UPPER_COUNT)]


def _add_upper(store: list, row: int, column: int, term) -> None:
    index = _upper_index(row, column)
    store[index] = store[index] + term


def _scale_upper(store: list, factor) -> list:
    return [entry * factor for entry in store]


def _sum_upper(left: list, right: list) -> list:
    return [a + b for a, b in zip(left, right, strict=True)]


def _dyadic_pair(pg, value) -> tuple[str, str]:
    lower = pg._driver_endpoint(value.lower())
    upper = pg._driver_endpoint(value.upper())
    if lower > upper:
        raise ArithmeticError("dyadic interval reversed")
    return str(lower), str(upper)


def _upper_to_payload(pg, store: list) -> dict[str, object]:
    return {
        "size": VARIABLES,
        "upper_triangle": [_dyadic_pair(pg, entry) for entry in store],
    }


def _contract(pg, store: list, coefficients: tuple) -> object:
    total = pg.arb(0)
    for row, left in enumerate(coefficients):
        for column in range(row, VARIABLES):
            factor = left * coefficients[column]
            if row != column:
                factor *= 2
            total = total + store[_upper_index(row, column)] * pg.rational(factor)
    return total


def _overlaps(left, right) -> bool:
    return left.lower() <= right.upper() and right.lower() <= left.upper()


def _require_overlap(label: str, contracted, scalar) -> None:
    if not _overlaps(contracted, scalar):
        raise ArithmeticError(
            f"{label} kernel contraction is disjoint from the scalar form: "
            f"contracted={contracted} scalar={scalar}"
        )
    print(
        "OVERLAP",
        label,
        "contracted",
        contracted,
        "scalar",
        scalar,
        flush=True,
    )


def _outward_rows(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = int(low.shape[1])
    if length == 0:
        return np.zeros(low.shape[0]), np.zeros(high.shape[0])
    unit = float(np.finfo(float).eps / 2)
    gamma = length * unit / (1.0 - length * unit)
    summed_low = low.sum(axis=1)
    summed_high = high.sum(axis=1)
    absolute_low = np.abs(low).sum(axis=1)
    absolute_high = np.abs(high).sum(axis=1)
    lower = np.nextafter(summed_low - gamma * absolute_low, -np.inf)
    upper = np.nextafter(summed_high + gamma * absolute_high, np.inf)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ArithmeticError("invalid batched directed sum")
    return lower, upper


def _multiply_rows(
    left_low: np.ndarray,
    left_high: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corners = (
        left_low * right_low,
        left_low * right_high,
        left_high * right_low,
        left_high * right_high,
    )
    return (
        np.nextafter(np.minimum.reduce(corners), -np.inf),
        np.nextafter(np.maximum.reduce(corners), np.inf),
    )


def _bilinear(
    pg,
    left_low: np.ndarray,
    left_high: np.ndarray,
    left_present: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
    right_present: np.ndarray,
    factors: list[tuple[np.ndarray, np.ndarray]],
    mask: np.ndarray | None,
) -> list[list]:
    """Sound Gram of two interval rows, matching nested interval products."""

    count = left_low.shape[1]
    selected = np.flatnonzero(mask) if mask is not None else np.arange(count)
    table = [[pg.arb(0) for _ in range(VARIABLES)] for _ in range(VARIABLES)]
    if selected.size == 0:
        return table
    active_left = np.flatnonzero(left_present)
    active_right = np.flatnonzero(right_present)
    if active_left.size == 0 or active_right.size == 0:
        return table
    sliced_factors = [(low[selected], high[selected]) for low, high in factors]
    right_rows_low = right_low[active_right][:, selected]
    right_rows_high = right_high[active_right][:, selected]
    for left_index in active_left:
        product_low, product_high = _multiply_rows(
            left_low[left_index, selected],
            left_high[left_index, selected],
            right_rows_low,
            right_rows_high,
        )
        for factor_low, factor_high in sliced_factors:
            product_low, product_high = _multiply_rows(
                product_low,
                product_high,
                factor_low,
                factor_high,
            )
        summed_low, summed_high = _outward_rows(product_low, product_high)
        for offset, right_index in enumerate(active_right):
            table[int(left_index)][int(right_index)] = pg.arb(
                float(summed_low[offset])
            ).union(pg.arb(float(summed_high[offset])))
    return table


def _accumulate_bilinear(pg, store: list, table: list[list], *, doubled: bool) -> None:
    half = pg.rational(pg.F(1, 2))
    for row in range(VARIABLES):
        for column in range(row, VARIABLES):
            symmetric = table[row][column] + table[column][row]
            term = symmetric if doubled else symmetric * half
            _add_upper(store, row, column, term)


def _remainings(pg) -> list[tuple[int, ...]]:
    found: set[tuple[int, ...]] = set()
    for signature in pg.CAP_SIGNATURES:
        for remaining, _exponent, _multiplicity in pg.fiber_splits(tuple(signature)):
            found.add(tuple(remaining))
    return sorted(found, key=lambda item: (len(item), item))


def _basis_path(destination: Path, basis: int) -> Path:
    return destination / f"basis_{basis:02d}.npz"


def _compute_basis(pg, engine, basis: int, destination: Path, index: dict) -> None:
    """Store one coefficient's shell affines without refitting another dimension."""

    path = _basis_path(destination, basis)
    if path.exists():
        print("BASIS_AFFINE_REUSED", basis, flush=True)
        return
    signature, degree = engine.descriptors[basis]
    original = engine.coefficient_fractions
    coefficients = [pg.F(0)] * VARIABLES
    coefficients[basis] = pg.F(1)
    try:
        engine.coefficient_fractions = tuple(coefficients)
        engine._affine = None
        engine._forms = None
        parts = engine.frozen_shell_affine(progress=False)
        shell_count = len(engine.outer_shells)
        if len(parts) != shell_count:
            raise ArithmeticError("affine shell count changed")
        low = np.zeros((shell_count, len(index), engine.n), dtype=np.float64)
        high = np.zeros_like(low)
        present = np.zeros((shell_count, len(index)), dtype=np.bool_)
        for shell_index, part in enumerate(parts):
            for remaining, values in part.items():
                key = tuple(remaining)
                if key not in index:
                    raise ArithmeticError("unexpected affine remaining signature")
                interval = pg.float_interval(values)
                slot = index[key]
                low[shell_index, slot] = interval[0]
                high[shell_index, slot] = interval[1]
                present[shell_index, slot] = True
    finally:
        engine.coefficient_fractions = original
        engine._affine = None
        engine._forms = None
    temporary = destination / f".basis_{basis:02d}.partial"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, low=low, high=high, present=present)
    temporary.replace(path)
    print("BASIS_AFFINE", basis, signature, degree, flush=True)


def _init_basis_worker(evaluator: str, source: bool) -> None:
    global _PG, _ENGINE, _BASIS_INDEX
    _PG, _transformed = _bind_evaluator(Path(evaluator))
    del _transformed
    _ENGINE = _PG._driver_engine(source=source)
    remainings = _remainings(_PG)
    _BASIS_INDEX = {signature: position for position, signature in enumerate(remainings)}


def _basis_worker(spec: tuple[int, str]) -> int:
    basis, destination = spec
    assert _PG is not None and _ENGINE is not None and _BASIS_INDEX is not None
    _compute_basis(_PG, _ENGINE, basis, Path(destination), _BASIS_INDEX)
    return basis


def _stack_basis_bank(destination: Path) -> None:
    sample = np.load(_basis_path(destination, 0))
    shell_count, remaining_count, length = sample["low"].shape
    del sample
    shape = (shell_count, remaining_count, VARIABLES, length)
    chunks = {
        "low": np.lib.format.open_memmap(
            destination / "low.npy.building",
            mode="w+",
            dtype=np.float64,
            shape=shape,
        ),
        "high": np.lib.format.open_memmap(
            destination / "high.npy.building",
            mode="w+",
            dtype=np.float64,
            shape=shape,
        ),
        "present": np.lib.format.open_memmap(
            destination / "present.npy.building",
            mode="w+",
            dtype=np.bool_,
            shape=shape[:-1],
        ),
    }
    for basis in range(VARIABLES):
        data = np.load(_basis_path(destination, basis))
        if data["low"].shape != (shell_count, remaining_count, length):
            raise ArithmeticError(f"basis {basis} affine shape changed")
        chunks["low"][:, :, basis, :] = data["low"]
        chunks["high"][:, :, basis, :] = data["high"]
        chunks["present"][:, :, basis] = data["present"]
        del data
    for array in chunks.values():
        array.flush()
    del chunks
    gc.collect()
    for name in ("low", "high", "present"):
        source = destination / f"{name}.npy.building"
        source.replace(destination / f"{name}.npy")
    print("AFFINE_BANK_STACKED", destination, flush=True)


def _build_affine_bank(
    evaluator: Path,
    destination: Path,
    *,
    source: bool,
    workers: int,
) -> None:
    if (destination / "low.npy").exists():
        print("AFFINE_BANK_REUSED", destination, flush=True)
        return
    pg, _transformed = _bind_evaluator(evaluator)
    del _transformed
    remainings = _remainings(pg)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "remainings.json").write_text(
        json.dumps(remainings),
        encoding="utf-8",
    )
    pending = [
        basis
        for basis in range(VARIABLES)
        if not _basis_path(destination, basis).exists()
    ]
    print("BASIS_PENDING", len(pending), "source" if source else "cap", flush=True)
    if pending:
        if workers == 1:
            engine = pg._driver_engine(source=source)
            index = {signature: position for position, signature in enumerate(remainings)}
            for basis in pending:
                _compute_basis(pg, engine, basis, destination, index)
            del engine
            gc.collect()
        else:
            del pg
            gc.collect()
            context = mp.get_context("spawn")
            jobs = [(basis, str(destination)) for basis in pending]
            with context.Pool(
                workers,
                initializer=_init_basis_worker,
                initargs=(str(evaluator), source),
            ) as pool:
                for basis in pool.imap_unordered(_basis_worker, jobs, chunksize=1):
                    print("BASIS_AFFINE_DONE", basis, flush=True)
    _stack_basis_bank(destination)


def _load_bank(path: Path) -> dict[str, object]:
    remainings = [tuple(item) for item in json.loads((path / "remainings.json").read_text())]
    return {
        "remainings": remainings,
        "index": {signature: position for position, signature in enumerate(remainings)},
        "low": np.load(path / "low.npy", mmap_mode="r"),
        "high": np.load(path / "high.npy", mmap_mode="r"),
        "present": np.load(path / "present.npy"),
    }


def _needed_degrees(descriptors) -> dict[tuple[int, ...], set[int]]:
    needed: dict[tuple[int, ...], set[int]] = {}
    for row, (left_signature, left_degree) in enumerate(descriptors):
        for right_signature, right_degree in descriptors[row:]:
            signature = tuple(sorted(tuple(left_signature) + tuple(right_signature)))
            needed.setdefault(signature, set()).add(int(left_degree) + int(right_degree))
    return needed


def _monomial_intervals(pg, engine) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    powers = {}
    for degree in range(13):
        values = tuple(point**degree for point in engine.full_radial)
        powers[degree] = pg.float_interval(values)
    return powers


def _monomial_cache(pg, engine):
    cached = getattr(engine, "_omnibias_monomials", None)
    if cached is None:
        cached = _monomial_intervals(pg, engine)
        engine._omnibias_monomials = cached
    return cached


def _cached_calls(builder):
    cache: dict = {}

    def wrapped(signature):
        hit = cache.get(signature)
        if hit is None:
            hit = builder(signature)
            cache[signature] = hit
        return hit

    return wrapped


def _outer_face_moments(
    pg,
    engine,
    contractions,
    jets,
    kind,
    count,
    radial,
    measures,
    source,
    fractions,
    minimum,
):
    """Build each erased fiber once, then cache the merged moment intervals."""

    if kind == "low":
        erased = pg._source_low_coordinates(engine, measures, 0)
        plain, witness = (
            pg._source_fiber(contractions, row, engine.fixed_bits, radial)
            for row in erased
        )

        def build(signature):
            return pg._interval_add(
                pg.interval_multiply(
                    witness,
                    jets.rows(count - 1, signature).binary64_intervals(),
                ),
                pg.interval_multiply(
                    plain,
                    jets.rows(count - 1, signature, marks="witness").binary64_intervals(),
                ),
            )

        return _cached_calls(build)
    if kind == "rank_two":
        erased = pg.tighten_palm_owner_kernels(
            pg.largest_palm_factory(engine, 0).kernels(source),
            engine,
            source,
            fractions=fractions,
        )
        plain, owner, difference, same = (
            pg._source_fiber(contractions, row, engine.fixed_bits)
            for row in (
                erased.ordinary,
                erased.owner,
                erased.nonowner_difference,
                erased.same_owner_difference,
            )
        )

        def build(signature):
            ordinary, first, second, both = (
                jets.rows(count - 1, signature, channel=channel).binary64_intervals()
                for channel in ("ordinary", "first", "second", "both")
            )
            return pg._interval_add(
                pg._interval_add(
                    pg.interval_multiply(same, ordinary),
                    pg.interval_multiply(plain, both),
                ),
                pg._interval_add(
                    pg.interval_multiply(owner, second),
                    pg.interval_multiply(difference, first),
                ),
            )

        return _cached_calls(build)
    if kind != "high":
        raise ValueError(f"unknown outer face kind {kind}")
    erased, _degree = pg.factorial_measures(
        engine,
        minimum=minimum,
        hard_cap=contractions.hard_cap,
        profile_power=0,
        degree=3,
    )
    fibers = tuple(
        pg._source_fiber(contractions, row, engine.fixed_bits) for row in erased
    )

    def build_high(signature):
        answer = None
        for degree in range(4):
            value = pg.interval_multiply(
                fibers[degree],
                jets.rows(count - 1, signature, order=3 - degree).binary64_intervals(),
            )
            answer = value if answer is None else pg._interval_add(answer, value)
        return answer

    return _cached_calls(build_high)


def _fill_square(
    pg,
    descriptors,
    kernels: dict[tuple[tuple[int, ...], int], object],
) -> list:
    store = _zero_upper(pg)
    for row, (left_signature, left_degree) in enumerate(descriptors):
        for column in range(row, VARIABLES):
            right_signature, right_degree = descriptors[column]
            signature = tuple(sorted(tuple(left_signature) + tuple(right_signature)))
            degree = int(left_degree) + int(right_degree)
            _add_upper(store, row, column, kernels[(signature, degree)])
    return store


def _denominator_matrix(pg, engine, descriptors, powers) -> list:
    needed = _needed_degrees(descriptors)
    kernels: dict[tuple[tuple[int, ...], int], object] = {}
    for shell, mask in zip(engine.outer_shells, engine.outer_masks, strict=True):
        for signature, degrees in needed.items():
            moments = engine.moment_interval(shell.ceiling, engine.k, signature)
            for degree in degrees:
                radial = powers[degree]
                contribution = pg.outward_sum(
                    pg.interval_multiply(radial, moments),
                    mask,
                )
                key = (signature, degree)
                kernels[key] = contribution if key not in kernels else kernels[key] + contribution
        engine.release_moment_caches()
        print("DENOMINATOR_SHELL", shell.ceiling, flush=True)
    return _fill_square(pg, descriptors, kernels)


def _layer_rows(bank: dict[str, object], shell_ids: list[int]):
    low = bank["low"]
    high = bank["high"]
    present = bank["present"]
    if not shell_ids:
        count = low.shape[1]
        return (
            np.zeros((count, VARIABLES, low.shape[-1])),
            np.zeros((count, VARIABLES, low.shape[-1])),
            np.zeros((count, VARIABLES), dtype=np.bool_),
        )
    first, *rest = shell_ids
    rows_low = np.array(low[first], copy=True)
    rows_high = np.array(high[first], copy=True)
    rows_present = np.array(present[first], copy=True)
    for shell_id in rest:
        _add_present_shell(
            rows_low,
            rows_high,
            rows_present,
            low[shell_id],
            high[shell_id],
            present[shell_id],
        )
    return rows_low, rows_high, rows_present


def _add_present_shell(
    rows_low: np.ndarray,
    rows_high: np.ndarray,
    rows_present: np.ndarray,
    shell_low: np.ndarray,
    shell_high: np.ndarray,
    shell_present: np.ndarray,
) -> None:
    """Add one shell without pushing an exact first contribution off its endpoints."""

    fresh = shell_present & ~rows_present
    both = shell_present & rows_present
    if np.any(fresh):
        rows_low[fresh] = shell_low[fresh]
        rows_high[fresh] = shell_high[fresh]
    if np.any(both):
        rows_low[both] = np.nextafter(rows_low[both] + shell_low[both], -np.inf)
        rows_high[both] = np.nextafter(rows_high[both] + shell_high[both], np.inf)
    rows_present |= shell_present


def _pair_store(pg, rows_low, rows_high, rows_present, remainings, factors_for, mask) -> list:
    store = _zero_upper(pg)
    available = [
        signature
        for signature, position in (
            (signature, position) for position, signature in enumerate(remainings)
        )
        if np.any(rows_present[position])
    ]
    ordered = sorted(available, key=lambda signature: (sum(signature), signature))
    positions = {signature: remainings.index(signature) for signature in ordered}
    for left_at, left in enumerate(ordered):
        for right in ordered[left_at:]:
            factors = factors_for(tuple(sorted(left + right)))
            table = _bilinear(
                pg,
                rows_low[positions[left]],
                rows_high[positions[left]],
                rows_present[positions[left]],
                rows_low[positions[right]],
                rows_high[positions[right]],
                rows_present[positions[right]],
                factors,
                mask,
            )
            _accumulate_bilinear(pg, store, table, doubled=left != right)
    return store


def _cap_regions(pg, engine, bank: dict[str, object]) -> dict[str, list]:
    remainings = bank["remainings"]
    assert isinstance(remainings, list)
    regions = {
        "J0": _zero_upper(pg),
        "Jplus": _zero_upper(pg),
        "Jtail": _zero_upper(pg),
    }
    names = ("J0", "Jplus", "Jtail")
    for layer, cap in enumerate(engine.caps[: engine.active_layers]):
        allowed = engine.inner_allowed[:, layer]
        masks = (
            allowed[0],
            allowed[1] & ~allowed[0],
            allowed[2] & ~allowed[1],
        )
        if not any(np.any(mask) for mask in masks):
            continue
        shell_ids = [
            position
            for position, shell in enumerate(engine.outer_shells)
            if pg.cap_leq(cap, shell.ceiling)
        ]
        rows_low, rows_high, rows_present = _layer_rows(bank, shell_ids)

        def factors_for(signature: tuple[int, ...], *, _layer=layer, _cap=cap) -> list:
            moments = engine.moment_interval(_cap, engine.k - 1, signature)
            if _layer:
                previous = engine.caps[_layer - 1]
                moments = pg._cap_positive_difference(
                    moments,
                    engine.moment_interval(previous, engine.k - 1, signature),
                )
            return [moments]

        for name, mask in zip(names, masks, strict=True):
            if not np.any(mask):
                continue
            contribution = _pair_store(
                pg,
                rows_low,
                rows_high,
                rows_present,
                remainings,
                factors_for,
                mask,
            )
            scaled = _scale_upper(contribution, engine.normalization)
            regions[name] = _sum_upper(regions[name], scaled)
        engine.release_moment_caches(retain=cap)
        print("CAP_LAYER", layer, cap, flush=True)
    engine.release_moment_caches()
    return regions


def _validate_cap(pg, engine, matrices: dict[str, list], coefficients) -> None:
    scalar = engine.form_scalars(progress=True)
    for name in ("denominator", "J0", "Jplus", "Jtail"):
        _require_overlap(name, _contract(pg, matrices[name], coefficients), scalar[name])


def _root_matrix(pg, engine, contractions, moment, radial, powers, descriptors) -> list:
    needed = _needed_degrees(descriptors)
    mask = np.zeros(engine.n, dtype=np.bool_)
    mask[np.asarray(contractions.indices, dtype=np.int64)] = True
    multiplier = None
    if radial is not None:
        multiplier = pg.float_interval(radial)
        np.maximum(multiplier[0], 0, out=multiplier[0])
    kernels: dict[tuple[tuple[int, ...], int], object] = {}
    for signature, degrees in needed.items():
        moments = moment(signature).binary64_intervals()
        for degree in degrees:
            radial_values = powers[degree]
            if multiplier is not None:
                radial_values = pg.interval_multiply(radial_values, multiplier)
            kernels[(signature, degree)] = pg.outward_sum(
                pg.interval_multiply(radial_values, moments),
                mask,
            )
    store = _fill_square(pg, descriptors, kernels)
    return _scale_upper(store, pg.rational(engine.k * contractions.maximum_weight))


def _prefix_store(
    pg,
    engine,
    bank,
    contractions,
    moment_intervals,
    *,
    largest_range,
    radial_factor,
) -> list:
    remainings = bank["remainings"]
    shells = engine.outer_shells
    low_bound, high_bound = (
        (pg.F(0), contractions.hard_cap)
        if largest_range is None
        else tuple(map(pg.F, largest_range))
    )
    high_bound = min(high_bound, contractions.hard_cap)
    shell_count = len(shells)
    running_low = np.zeros((len(remainings), VARIABLES, engine.n))
    running_high = np.zeros_like(running_low)
    running_present = np.zeros((len(remainings), VARIABLES), dtype=np.bool_)
    total = _zero_upper(pg)
    outer = contractions.role == "outer"
    weights = pg.float_interval(contractions.face_weights) if outer else None
    mask = None
    if not outer:
        mask = np.zeros(engine.n, dtype=np.bool_)
        mask[np.asarray(contractions.indices, dtype=np.int64)] = True
    for length, shell in enumerate(shells, start=1):
        _add_present_shell(
            running_low,
            running_high,
            running_present,
            bank["low"][length - 1],
            bank["high"][length - 1],
            bank["present"][length - 1],
        )
        current = high_bound if shell.ceiling is None else min(high_bound, pg.F(shell.ceiling))
        if length == shell_count:
            following = pg.F(0)
        elif shells[length].ceiling is not None:
            following = pg.F(shells[length].ceiling)
        else:
            following = high_bound
        possible = (
            (current > low_bound if largest_range is not None else current >= 0)
            if length == shell_count
            else max(low_bound, following) < current
        )
        if not possible:
            continue

        def factors_for(signature: tuple[int, ...], *, _weights=weights, _radial=radial_factor):
            moments = moment_intervals(signature)
            factors = []
            if _weights is not None:
                factors.append(_weights)
            if _radial is not None:
                factors.append(_radial)
            factors.append(moments)
            return factors

        total = _sum_upper(
            total,
            _pair_store(
                pg,
                running_low,
                running_high,
                running_present,
                remainings,
                factors_for,
                mask,
            ),
        )
    factor = pg.arb(engine.k) * engine.h * (engine.h if outer else 1) / engine.Z
    return _scale_upper(total, factor)


def _component_matrices(pg, engine, bank, task: dict) -> dict[str, list]:
    group = pg.GROUP_BY_ID[task["group"]]
    kind = task["kind"]
    parameters = task["parameters"]
    original = group
    if kind == "low":
        group = pg.source_clipped_group(original, engine.hq, parameters["high"])
    outer = original["role"] == "outer"
    zero_names = (
        ("root_square", "outer_face_square") if outer else ("inner_face",)
    )
    if group is None or (
        kind == "high" and pg.F(group["split"]) >= pg.F(group["hard_cap"])
    ):
        return {name: _zero_upper(pg) for name in zero_names}
    contractions = pg.SourceContractions(
        engine,
        group,
        pg.DERIVED_INPUTS["hybrid"]["outer_absolute_weights"],
    )
    count = contractions.dimension
    radial = None
    inner_radial = None
    largest_range = None
    multiplier = pg.arb(1)
    if kind == "low":
        low, high, slope = (pg.F(parameters[name]) for name in ("low", "high", "slope"))
        measures = pg.source_low_measures(
            engine,
            ceiling=group["ceiling"],
            order=group["order"],
            hard_cap=group["hard_cap"],
            low=low,
            high=high,
            slope=slope,
        )
        ordinary, witness = pg._source_low_coordinates(engine, measures, 2)
        jets = pg.SourceJets(
            ordinary,
            engine.midpoints,
            engine.fixed_bits,
            marks={"witness": witness},
        )
        offset = (pg.F(group["order"]) - 1) * measures.high + count * engine.hq
        active = set(contractions.indices)
        rate = pg.rational(slope)
        ceiling = pg.F(group["ceiling"])
        radial = tuple(
            (rate * pg.rational(point * engine.hq - ceiling + offset)).exp()
            if point in active
            else pg.arb(0)
            for point in range(engine.n)
        )
        if outer:
            radial = tuple(value.upper() for value in radial)
            selected = lambda dimension, signature: jets.rows(
                dimension, signature, marks="witness"
            )
        else:
            inner_radial = pg.float_interval(radial)
            np.maximum(inner_radial[0], 0, out=inner_radial[0])
            inactive = np.ones(engine.n, dtype=np.bool_)
            inactive[np.asarray(contractions.indices, dtype=np.int64)] = False
            inner_radial[0][inactive] = 0.0
            inner_radial[1][inactive] = 0.0
            radial = None
            selected = lambda dimension, signature: jets.rows(
                dimension, signature, marks="witness"
            )
    elif kind == "rank_two":
        low, high = pg.F(parameters["q_low"]), pg.F(parameters["q_high"])
        source = pg.LargestFragmentBin(
            low,
            high,
            pg.F(group["ceiling"]),
            pg.F(group["order"]),
        )
        fractions = pg.owner_feasible_mass_fractions(engine, source)
        retained = pg.tighten_palm_owner_kernels(
            pg.largest_palm_factory(engine, 2).kernels(source),
            engine,
            source,
            fractions=fractions,
        )
        jets = pg.SourceJets(
            (
                retained.ordinary,
                retained.owner,
                retained.nonowner_difference,
                retained.same_owner_difference,
            ),
            engine.midpoints,
            engine.fixed_bits,
            ring="palm",
        )
        selected = lambda dimension, signature: jets.rows(dimension, signature, channel="both")
        multiplier = source.mass
        largest_range = None if outer else (low, high)
    elif kind == "high":
        retained, _degree = pg.factorial_measures(
            engine,
            minimum=pg.F(group["split"]),
            hard_cap=contractions.hard_cap,
            profile_power=2,
            degree=3,
        )
        jets = pg.SourceJets(
            retained,
            engine.midpoints,
            engine.fixed_bits,
            ring="factorial",
        )
        selected = lambda dimension, signature: jets.rows(dimension, signature, order=3)
    else:
        raise ValueError(f"unknown source kind {kind}")

    if not outer:
        def face_moments(signature):
            return selected(count, signature).binary64_intervals()

        face = _prefix_store(
            pg,
            engine,
            bank,
            contractions,
            face_moments,
            largest_range=largest_range,
            radial_factor=inner_radial,
        )
        if kind == "rank_two":
            face = _scale_upper(face, multiplier)
        scalar = pg._source_face_square(
            contractions,
            lambda signature: selected(count, signature).binary64_intervals(),
            largest_range=largest_range,
            radial_factor=inner_radial,
        )
        if kind == "rank_two":
            scalar = multiplier * scalar
        scalar = pg.positive_enclosure(scalar)
        _require_overlap(
            f"{task['group']}:{kind}:{task['index']}:inner_face",
            _contract(pg, face, engine.coefficient_fractions),
            scalar,
        )
        return {"inner_face": face}

    root = _root_matrix(
        pg,
        engine,
        contractions,
        lambda signature: selected(count, signature),
        radial,
        _monomial_cache(pg, engine),
        tuple(engine.descriptors),
    )
    scalar_root = pg._source_root_square(
        contractions,
        lambda signature: selected(count, signature),
        radial,
    )
    jets.release_count(count)
    face_moments = _outer_face_moments(
        pg,
        engine,
        contractions,
        jets,
        kind,
        count,
        radial,
        measures if kind == "low" else None,
        source if kind == "rank_two" else None,
        fractions if kind == "rank_two" else None,
        pg.F(group["split"]) if kind == "high" else None,
    )
    face = _prefix_store(
        pg,
        engine,
        bank,
        contractions,
        face_moments,
        largest_range=None,
        radial_factor=None,
    )
    scalar_face = pg._source_face_square(contractions, face_moments)
    if kind == "rank_two":
        root = _scale_upper(root, multiplier)
        face = _scale_upper(face, multiplier)
        scalar_root = multiplier * scalar_root
        scalar_face = multiplier * scalar_face
    scalar_root = pg.positive_enclosure(scalar_root)
    scalar_face = pg.positive_enclosure(scalar_face)
    label = f"{task['group']}:{kind}:{task['index']}"
    _require_overlap(
        label + ":root_square",
        _contract(pg, root, engine.coefficient_fractions),
        scalar_root,
    )
    _require_overlap(
        label + ":outer_face_square",
        _contract(pg, face, engine.coefficient_fractions),
        scalar_face,
    )
    return {"root_square": root, "outer_face_square": face}


def _weight_component(pg, matrices: dict[str, list], task: dict) -> list:
    role = pg.GROUP_BY_ID[task["group"]]["role"]
    if role == "outer":
        young = pg.F(task["young_q"], task["young_denominator"])
        root = _scale_upper(matrices["root_square"], pg.rational(young))
        face = _scale_upper(
            matrices["outer_face_square"],
            pg.rational(1 / young),
        )
        return _sum_upper(root, face)
    coefficient = pg.F(task["restoration_coefficient"])
    return _scale_upper(matrices["inner_face"], pg.rational(coefficient))


def _init_worker(evaluator: str, bank_dir: str) -> None:
    global _PG, _ENGINE, _BANK
    _PG, _transformed = _bind_evaluator(Path(evaluator))
    del _transformed
    _ENGINE = _PG._driver_engine(source=True)
    _BANK = _load_bank(Path(bank_dir))


def _worker(task: dict) -> tuple[str, list[list[str]]]:
    assert _PG is not None and _ENGINE is not None and _BANK is not None
    matrices = _component_matrices(_PG, _ENGINE, _BANK, task)
    weighted = _weight_component(_PG, matrices, task)
    payload = [list(_dyadic_pair(_PG, entry)) for entry in weighted]
    key = f"{task['group']}__{task['kind']}__{task['index']}"
    print("SOURCE_KERNEL", key, flush=True)
    return key, payload


def _sum_dyadic_rows(rows: list[list[list[str]]]) -> list[list[str]]:
    if not rows:
        raise ArithmeticError("source kernel inventory is empty")
    total = [Fraction(0) for _ in range(UPPER_COUNT)]
    upper = [Fraction(0) for _ in range(UPPER_COUNT)]
    for row in rows:
        if len(row) != UPPER_COUNT:
            raise ArithmeticError("source kernel row has the wrong length")
        for index, endpoints in enumerate(row):
            low, high = Fraction(endpoints[0]), Fraction(endpoints[1])
            if low > high:
                raise ArithmeticError("source kernel endpoint order changed")
            total[index] += low
            upper[index] += high
    return [[str(low), str(high)] for low, high in zip(total, upper, strict=True)]


def _descriptors(pg) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "signature": list(signature),
            "radial_degree": degree,
        }
        for index, (signature, degree) in enumerate(
            (signature, degree)
            for signature in pg.CAP_SIGNATURES
            for degree in range(7)
        )
    ]


def _task_keys(pg) -> list[list[object]]:
    return [[task["group"], task["kind"], int(task["index"])] for task in pg.TASKS]


def _write_receipt(
    transformed: str,
    pg,
    matrices: dict[str, dict[str, object]],
    output: Path,
) -> None:
    receipt = {
        "schema": "omnibias.prime_gap_quadratic.v1",
        "algorithm": "coefficient_independent_interval_gram_v1",
        "source_sha256": PINNED_SOURCE_SHA256,
        "transformed_source_sha256": transformed,
        "dimension": 39,
        "intervals": 98304,
        "arb_precision_bits": 160,
        "cap_fractional_bits": 224,
        "source_fractional_bits": 192,
        "signed_convolution_strategy": "nonnegative_part_split",
        "descriptors": _descriptors(pg),
        "source_task_keys": _task_keys(pg),
        "matrices": matrices,
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"wrote {output}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    pg, transformed = _bind_evaluator(args.evaluator)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    cap_path = work / "cap_matrices.json"
    if not cap_path.exists():
        bank_dir = work / "cap_bank"
        print("CAP_BANK_START", flush=True)
        _build_affine_bank(
            args.evaluator,
            bank_dir,
            source=False,
            workers=args.workers,
        )
        engine = pg._driver_engine(source=False)
        bank = _load_bank(bank_dir)
        descriptors = tuple(engine.descriptors)
        powers = _monomial_intervals(pg, engine)
        print("DENOMINATOR_START", flush=True)
        denominator = _denominator_matrix(pg, engine, descriptors, powers)
        print("CAP_REGIONS_START", flush=True)
        regions = _cap_regions(pg, engine, bank)
        matrices = {"denominator": denominator, **regions}
        _validate_cap(pg, engine, matrices, engine.coefficient_fractions)
        payload = {
            "transformed_source_sha256": transformed,
            "matrices": {
                name: _upper_to_payload(pg, store) for name, store in matrices.items()
            },
        }
        cap_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        del engine, bank, denominator, regions, matrices
        gc.collect()
        print("CAP_KERNELS_VALIDATED", flush=True)
    else:
        print("CAP_KERNELS_REUSED", flush=True)

    source_bank = work / "source_bank"
    if not (source_bank / "low.npy").exists():
        print("SOURCE_BANK_START", flush=True)
        _build_affine_bank(
            args.evaluator,
            source_bank,
            source=True,
            workers=args.workers,
        )
    done_dir = work / "source_rows"
    done_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for task in pg.TASKS:
        key = f"{task['group']}__{task['kind']}__{task['index']}"
        if not (done_dir / f"{key}.json").exists():
            pending.append(task)
    print("SOURCE_PENDING", len(pending), "of", len(pg.TASKS), flush=True)
    if pending:
        context = mp.get_context("spawn")
        with context.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(str(args.evaluator), str(source_bank)),
        ) as pool:
            for key, row in pool.imap_unordered(_worker, pending, chunksize=1):
                target = done_dir / f"{key}.json"
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(row), encoding="utf-8")
                temporary.replace(target)
                print("SOURCE_KERNEL_STORED", key, flush=True)
    rows = []
    for task in pg.TASKS:
        key = f"{task['group']}__{task['kind']}__{task['index']}"
        path = done_dir / f"{key}.json"
        if not path.exists():
            raise ArithmeticError(f"missing source kernel {key}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) != 97:
        raise ArithmeticError("source kernel inventory is not the complete 97 tasks")
    cap_payload = json.loads(cap_path.read_text(encoding="utf-8"))
    if cap_payload["transformed_source_sha256"] != transformed:
        raise ArithmeticError("cap kernel checkpoint is from a different evaluator")
    matrices = dict(cap_payload["matrices"])
    matrices["source_loss"] = {
        "size": VARIABLES,
        "upper_triangle": _sum_dyadic_rows(rows),
    }
    expected = {"denominator", "J0", "Jplus", "Jtail", "source_loss"}
    if set(matrices) != expected:
        raise ArithmeticError("quadratic matrix inventory is incomplete")
    _write_receipt(transformed, pg, matrices, args.output)


if __name__ == "__main__":
    main()
