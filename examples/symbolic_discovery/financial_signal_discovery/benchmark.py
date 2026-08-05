# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Milestone-gated calculus-native financial signal discovery benchmark.

Milestone 1 is intentionally offline and reproducible. It asks a narrow
question: can local Taylor/jet features extracted from a volatility history add
forecasting value over a standard HAR-style realized-volatility baseline?

The synthetic process is not meant to be a market claim. It creates a controlled
regime where volatility has smooth local dynamics, so slope/curvature features
should be useful. Later milestones should replace this with real walk-forward
data and transaction-cost-aware objectives.
"""

from __future__ import annotations

import ast
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np


@dataclass(frozen=True)
class VolatilityDataset:
    returns: np.ndarray
    latent_vol: np.ndarray
    observed_vol: np.ndarray
    target_vol: np.ndarray
    feature_start: int
    horizon: int


@dataclass(frozen=True)
class MarketOHLCV:
    symbol: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    adj_close: np.ndarray
    volume: np.ndarray


@dataclass(frozen=True)
class ChronologicalSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    names: list[str]


@dataclass(frozen=True)
class FittedLinearModel:
    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    target_transform: str = "identity"
    prediction_scale: float = 1.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=float) - self.x_mean) / self.x_scale
        pred = self.intercept + xs @ self.coefficients
        if self.target_transform == "log":
            return self.prediction_scale * np.exp(pred)
        return self.prediction_scale * pred

    def selected_features(self, *, top_k: int = 12) -> list[dict[str, float | str]]:
        scale = np.where(self.x_scale < 1e-12, 1.0, self.x_scale)
        raw_coef = self.coefficients / scale
        order = np.argsort(-np.abs(raw_coef))[:top_k]
        return [
            {"name": self.feature_names[index], "coefficient": float(raw_coef[index])}
            for index in order
            if abs(raw_coef[index]) > 1e-12
        ]


def make_synthetic_volatility_dataset(
    *,
    n_days: int = 2600,
    horizon: int = 5,
    seed: int = 0,
) -> VolatilityDataset:
    """Generate a smooth stochastic-volatility process with predictable jets."""

    rng = np.random.default_rng(seed)
    log_vol = np.zeros(n_days + horizon + 30, dtype=float)
    log_vol[0] = np.log(0.018)
    latent_cycle = rng.uniform(0.0, 2.0 * np.pi)
    for t in range(1, log_vol.size):
        seasonal = 0.06 * np.sin(2.0 * np.pi * t / 126.0 + latent_cycle)
        medium_cycle = 0.04 * np.sin(2.0 * np.pi * t / 31.0)
        shock = rng.normal(0.0, 0.025)
        log_vol[t] = 0.982 * log_vol[t - 1] + 0.018 * np.log(0.018) + seasonal + medium_cycle + shock

    latent_vol = np.exp(log_vol)
    returns = latent_vol * rng.normal(0.0, 1.0, size=latent_vol.size)
    observed_vol = _rolling_rms(returns, window=5)
    target_vol = np.asarray(
        [
            np.sqrt(np.mean(returns[t + 1 : t + horizon + 1] ** 2))
            if t + horizon < returns.size
            else np.nan
            for t in range(returns.size)
        ],
        dtype=float,
    )
    return VolatilityDataset(
        returns=returns,
        latent_vol=latent_vol,
        observed_vol=observed_vol,
        target_vol=target_vol,
        feature_start=30,
        horizon=horizon,
    )


def build_har_features(dataset: VolatilityDataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    targets = []
    for t in _valid_feature_times(dataset):
        vol = dataset.observed_vol
        rows.append(
            [
                vol[t],
                float(np.mean(vol[t - 4 : t + 1])),
                float(np.mean(vol[t - 21 : t + 1])),
            ]
        )
        targets.append(dataset.target_vol[t])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), ["rv_1", "rv_5", "rv_22"]


def build_jet_features(dataset: VolatilityDataset, *, lookback: int = 22) -> tuple[np.ndarray, np.ndarray, list[str]]:
    har, targets, har_names = build_har_features(dataset)
    rows = []
    vol = np.log(np.maximum(dataset.observed_vol, 1e-8))
    times = list(_valid_feature_times(dataset))
    for t in times:
        window = vol[t - lookback + 1 : t + 1]
        jets = _local_taylor_jets(window)
        rows.append(jets)
    jet_names = ["log_rv_level", "log_rv_slope", "log_rv_curvature", "log_rv_cubic", "log_rv_roughness"]
    jet_matrix = np.asarray(rows, dtype=float)
    return np.concatenate([har, jet_matrix], axis=1), targets, [*har_names, *jet_names]


def build_multiscale_jet_features(
    dataset: VolatilityDataset,
    *,
    lookbacks: tuple[int, ...] = (11, 22, 63),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    har, targets, har_names = build_har_features(dataset)
    rows = []
    vol = np.log(np.maximum(dataset.observed_vol, 1e-8))
    for t in _valid_feature_times(dataset):
        features = []
        for lookback in lookbacks:
            window = vol[t - lookback + 1 : t + 1]
            features.extend(_local_taylor_jets(window))
        rows.append(features)
    jet_names = [
        f"log_rv_{lookback}_{name}"
        for lookback in lookbacks
        for name in ("level", "slope", "curvature", "cubic", "roughness")
    ]
    return np.concatenate([har, np.asarray(rows, dtype=float)], axis=1), targets, [*har_names, *jet_names]


def evaluate_milestone1(
    *,
    n_days: int = 2600,
    horizon: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    dataset = make_synthetic_volatility_dataset(n_days=n_days, horizon=horizon, seed=seed)
    har = _chronological_split(*build_har_features(dataset))
    jets = _chronological_split(*build_jet_features(dataset))

    har_model = _fit_tuned_linear(har)
    jet_model = _fit_tuned_linear(jets)
    har_pred = har_model.predict(har.x_test)
    jet_pred = jet_model.predict(jets.x_test)
    har_metrics = _forecast_metrics(har.y_test, har_pred)
    jet_metrics = _forecast_metrics(jets.y_test, jet_pred)
    improvement = (har_metrics["rmse"] - jet_metrics["rmse"]) / har_metrics["rmse"]
    gate_passed = bool(improvement >= 0.02 and jet_metrics["qlike"] <= har_metrics["qlike"])
    return {
        "milestone": "M1 offline volatility jets",
        "claim": "local Taylor/jet features improve volatility forecasting over HAR on controlled data",
        "dataset": {
            "kind": "synthetic smooth stochastic volatility",
            "n_days": n_days,
            "horizon": horizon,
            "seed": seed,
            "target": "next-horizon realized volatility",
        },
        "models": {
            "har_baseline": har_metrics | {"alpha": har_model.alpha, "features": har.names},
            "jet_augmented": jet_metrics
            | {
                "alpha": jet_model.alpha,
                "features": jets.names,
                "selected_features": jet_model.selected_features(),
            },
        },
        "success_gate": {
            "rmse_improvement_required": 0.02,
            "rmse_improvement": improvement,
            "qlike_not_worse": jet_metrics["qlike"] <= har_metrics["qlike"],
            "passed": gate_passed,
        },
    }


def fetch_yahoo_ohlcv(
    *,
    symbol: str = "SPY",
    start: str = "2005-01-01",
    end: str = "2025-12-31",
    cache_dir: Path = Path("data/financial_signal_discovery"),
    refresh: bool = False,
    timeout: int = 30,
) -> MarketOHLCV:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol.upper()}_{start}_{end}_yahoo_chart.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text())
    else:
        url = _yahoo_chart_url(symbol=symbol, start=start, end=end)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cache_path.write_text(json.dumps(payload))
    return _parse_yahoo_chart(symbol=symbol, payload=payload)


def market_volatility_dataset_from_ohlcv(
    ohlcv: MarketOHLCV,
    *,
    horizon: int = 5,
    realized_window: int = 5,
) -> VolatilityDataset:
    close = np.maximum(ohlcv.adj_close, 1e-12)
    returns = np.diff(np.log(close))
    observed_vol = _rolling_rms(returns, window=realized_window) * np.sqrt(252.0)
    target_vol = np.asarray(
        [
            np.sqrt(np.mean(returns[t + 1 : t + horizon + 1] ** 2)) * np.sqrt(252.0)
            if t + horizon < returns.size
            else np.nan
            for t in range(returns.size)
        ],
        dtype=float,
    )
    return VolatilityDataset(
        returns=returns,
        latent_vol=observed_vol,
        observed_vol=observed_vol,
        target_vol=target_vol,
        feature_start=90,
        horizon=horizon,
    )


def evaluate_milestone2(
    *,
    symbol: str = "SPY",
    start: str = "2005-01-01",
    end: str = "2025-12-31",
    horizon: int = 5,
    cache_dir: Path = Path("data/financial_signal_discovery"),
    refresh: bool = False,
) -> dict[str, object]:
    ohlcv = fetch_yahoo_ohlcv(
        symbol=symbol,
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    dataset = market_volatility_dataset_from_ohlcv(ohlcv, horizon=horizon)
    har = _chronological_split(*build_har_features(dataset))
    jets = _chronological_split(*build_multiscale_jet_features(dataset))

    har_model = _fit_tuned_linear(har)
    jet_model = _fit_tuned_linear(jets)
    har_metrics = _forecast_metrics(har.y_test, har_model.predict(har.x_test))
    jet_metrics = _forecast_metrics(jets.y_test, jet_model.predict(jets.x_test))
    improvement = (har_metrics["rmse"] - jet_metrics["rmse"]) / har_metrics["rmse"]
    qlike_improvement = (har_metrics["qlike"] - jet_metrics["qlike"]) / abs(har_metrics["qlike"])
    gate_passed = bool(improvement >= 0.01 and jet_metrics["qlike"] <= har_metrics["qlike"])
    return {
        "milestone": "M2 real SPY volatility jets",
        "claim": "multiscale local volatility jets improve real walk-forward volatility forecasting over HAR",
        "dataset": {
            "kind": "Yahoo daily OHLCV",
            "symbol": ohlcv.symbol,
            "start": start,
            "end": end,
            "n_returns": int(dataset.returns.size),
            "horizon": horizon,
            "target": "next-horizon annualized realized volatility",
        },
        "models": {
            "har_baseline": har_metrics | {"alpha": har_model.alpha, "features": har.names},
            "jet_augmented": jet_metrics
            | {
                "alpha": jet_model.alpha,
                "features": jets.names,
                "selected_features": jet_model.selected_features(),
            },
        },
        "success_gate": {
            "rmse_improvement_required": 0.01,
            "rmse_improvement": improvement,
            "qlike_improvement": qlike_improvement,
            "qlike_not_worse": jet_metrics["qlike"] <= har_metrics["qlike"],
            "passed": gate_passed,
        },
        "analysis": _gate_analysis(
            baseline_name="HAR",
            candidate_name="multiscale volatility jets",
            baseline_metrics=har_metrics,
            candidate_metrics=jet_metrics,
            passed=gate_passed,
            improvement=improvement,
        ),
    }


def evaluate_milestone3(
    *,
    target_symbol: str = "SPY",
    context_symbols: tuple[str, ...] = ("^VIX", "QQQ", "IWM", "TLT", "GLD"),
    start: str = "2005-01-01",
    end: str = "2025-12-31",
    horizon: int = 5,
    cache_dir: Path = Path("data/financial_signal_discovery"),
    refresh: bool = False,
) -> dict[str, object]:
    symbols = (target_symbol, *context_symbols)
    markets = [
        fetch_yahoo_ohlcv(
            symbol=symbol,
            start=start,
            end=end,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        for symbol in symbols
    ]
    aligned = align_market_returns(markets)
    aligned_levels = align_market_adj_close(markets)
    target_returns = aligned[:, 0]
    target_dataset = volatility_dataset_from_returns(target_returns, horizon=horizon, feature_start=90)
    baseline = _chronological_split(*build_multiscale_jet_features(target_dataset))
    candidate = _chronological_split(
        *build_cross_asset_features(
            returns=aligned,
            levels=aligned_levels,
            symbols=tuple(market.symbol for market in markets),
            target_index=0,
            horizon=horizon,
        )
    )

    baseline_model = _fit_tuned_linear(baseline)
    candidate_model = _fit_tuned_linear(candidate)
    baseline_metrics = _forecast_metrics(baseline.y_test, baseline_model.predict(baseline.x_test))
    candidate_metrics = _forecast_metrics(candidate.y_test, candidate_model.predict(candidate.x_test))
    improvement = (baseline_metrics["rmse"] - candidate_metrics["rmse"]) / baseline_metrics["rmse"]
    qlike_improvement = (baseline_metrics["qlike"] - candidate_metrics["qlike"]) / abs(
        baseline_metrics["qlike"]
    )
    gate_passed = bool(improvement >= 0.01 and candidate_metrics["qlike"] <= baseline_metrics["qlike"])
    return {
        "milestone": "M3 cross-asset volatility context",
        "claim": "cross-asset volatility context improves SPY forecasting over SPY-only multiscale jets",
        "dataset": {
            "kind": "Yahoo daily OHLCV aligned close-to-close returns",
            "target_symbol": target_symbol.upper(),
            "context_symbols": [symbol.upper() for symbol in context_symbols],
            "start": start,
            "end": end,
            "n_aligned_returns": int(aligned.shape[0]),
            "horizon": horizon,
            "target": "target-symbol next-horizon annualized realized volatility",
        },
        "models": {
            "target_only_jets": baseline_metrics
            | {"alpha": baseline_model.alpha, "features": baseline.names},
            "cross_asset_jets": candidate_metrics
            | {
                "alpha": candidate_model.alpha,
                "features": candidate.names,
                "selected_features": candidate_model.selected_features(),
            },
        },
        "success_gate": {
            "rmse_improvement_required": 0.01,
            "rmse_improvement": improvement,
            "qlike_improvement": qlike_improvement,
            "qlike_not_worse": candidate_metrics["qlike"] <= baseline_metrics["qlike"],
            "passed": gate_passed,
        },
        "analysis": _gate_analysis(
            baseline_name="SPY-only jets",
            candidate_name="cross-asset jets",
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            passed=gate_passed,
            improvement=improvement,
        ),
    }


def evaluate_milestone4(
    *,
    target_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    context_universe: tuple[str, ...] = ("^VIX", "SPY", "QQQ", "IWM", "TLT", "GLD"),
    start: str = "2005-01-01",
    end: str = "2025-12-31",
    horizon: int = 5,
    cache_dir: Path = Path("data/financial_signal_discovery"),
    refresh: bool = False,
) -> dict[str, object]:
    target_results = []
    for target_symbol in target_symbols:
        context_symbols = tuple(
            symbol for symbol in context_universe if symbol.upper() != target_symbol.upper()
        )
        result = evaluate_milestone3(
            target_symbol=target_symbol,
            context_symbols=context_symbols,
            start=start,
            end=end,
            horizon=horizon,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        gate = result["success_gate"]
        models = result["models"]
        target_results.append(
            {
                "target_symbol": target_symbol.upper(),
                "context_symbols": [symbol.upper() for symbol in context_symbols],
                "baseline_rmse": models["target_only_jets"]["rmse"],
                "candidate_rmse": models["cross_asset_jets"]["rmse"],
                "baseline_qlike": models["target_only_jets"]["qlike"],
                "candidate_qlike": models["cross_asset_jets"]["qlike"],
                "rmse_improvement": gate["rmse_improvement"],
                "qlike_improvement": gate["qlike_improvement"],
                "passed": gate["passed"],
            }
        )

    improvements = np.asarray([row["rmse_improvement"] for row in target_results], dtype=float)
    pass_rate = float(np.mean([row["passed"] for row in target_results]))
    median_improvement = float(np.median(improvements))
    gate_passed = bool(pass_rate >= 2.0 / 3.0 and median_improvement >= 0.01)
    return {
        "milestone": "M4 multi-target robustness",
        "claim": "cross-asset volatility jets improve more than one equity target, not only SPY",
        "dataset": {
            "kind": "Yahoo daily OHLCV aligned close-to-close returns",
            "target_symbols": [symbol.upper() for symbol in target_symbols],
            "context_universe": [symbol.upper() for symbol in context_universe],
            "start": start,
            "end": end,
            "horizon": horizon,
        },
        "targets": target_results,
        "success_gate": {
            "target_pass_rate_required": 2.0 / 3.0,
            "target_pass_rate": pass_rate,
            "median_rmse_improvement_required": 0.01,
            "median_rmse_improvement": median_improvement,
            "passed": gate_passed,
        },
        "analysis": _robustness_analysis(target_results=target_results, passed=gate_passed),
    }


def evaluate_milestone5(
    *,
    target_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    context_universe: tuple[str, ...] = ("^VIX", "SPY", "QQQ", "IWM", "TLT", "GLD"),
    start: str = "2005-01-01",
    end: str = "2025-12-31",
    horizon: int = 5,
    cache_dir: Path = Path("data/financial_signal_discovery"),
    refresh: bool = False,
) -> dict[str, object]:
    """Compare calculus features against a stronger daily-data model zoo.

    This is not a universal SOTA claim. It is the next honest daily-OHLCV test:
    candidate models must beat the best baseline available from HAR, target-only
    jets, cross-asset non-jet context, and optional boosted-tree regressors.
    """

    target_results = []
    for target_symbol in target_symbols:
        context_symbols = tuple(
            symbol for symbol in context_universe if symbol.upper() != target_symbol.upper()
        )
        markets = [
            fetch_yahoo_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                cache_dir=cache_dir,
                refresh=refresh,
            )
            for symbol in (target_symbol, *context_symbols)
        ]
        aligned_returns = align_market_returns(markets)
        aligned_levels = align_market_adj_close(markets)
        symbols = tuple(market.symbol for market in markets)
        target_dataset = volatility_dataset_from_returns(aligned_returns[:, 0], horizon=horizon)

        baseline_splits = {
            "har_ridge": _chronological_split(*build_har_features(target_dataset)),
            "target_jets_model_zoo": _chronological_split(
                *build_multiscale_jet_features(target_dataset)
            ),
            "cross_asset_nonjet_model_zoo": _chronological_split(
                *build_cross_asset_standard_features(
                    returns=aligned_returns,
                    levels=aligned_levels,
                    symbols=symbols,
                    target_index=0,
                    horizon=horizon,
                )
            ),
        }
        candidate_split = _chronological_split(
            *build_cross_asset_sota_candidate_features(
                returns=aligned_returns,
                levels=aligned_levels,
                symbols=symbols,
                target_index=0,
                horizon=horizon,
            )
        )

        baseline_results = []
        baseline_results.extend(_evaluate_linear_only(baseline_splits["har_ridge"], "har_ridge"))
        baseline_results.extend(
            _evaluate_model_zoo(
                baseline_splits["target_jets_model_zoo"],
                "target_jets",
                include_boosters=True,
            )
        )
        baseline_results.extend(
            _evaluate_model_zoo(
                baseline_splits["cross_asset_nonjet_model_zoo"],
                "cross_asset_nonjet",
                include_boosters=True,
            )
        )
        candidate_results = _evaluate_model_zoo(
            candidate_split,
            "cross_asset_jets",
            include_boosters=True,
        )
        compact_candidate = _chronological_split(
            *build_cross_asset_compact_calculus_features(
                returns=aligned_returns,
                levels=aligned_levels,
                symbols=symbols,
                target_index=0,
                horizon=horizon,
            )
        )
        candidate_results.extend(
            _evaluate_model_zoo(
                compact_candidate,
                "cross_asset_compact_roughness_jets",
                include_boosters=True,
            )
        )

        best_baseline = _best_model_result(baseline_results)
        best_candidate = _best_model_result(candidate_results)
        improvement = (best_baseline["rmse"] - best_candidate["rmse"]) / best_baseline["rmse"]
        qlike_improvement = (best_baseline["qlike"] - best_candidate["qlike"]) / abs(
            best_baseline["qlike"]
        )
        passed = bool(improvement >= 0.01 and best_candidate["qlike"] <= best_baseline["qlike"])
        target_results.append(
            {
                "target_symbol": target_symbol.upper(),
                "context_symbols": [symbol.upper() for symbol in context_symbols],
                "best_baseline": best_baseline,
                "best_candidate": best_candidate,
                "baseline_rmse": best_baseline["rmse"],
                "candidate_rmse": best_candidate["rmse"],
                "baseline_qlike": best_baseline["qlike"],
                "candidate_qlike": best_candidate["qlike"],
                "rmse_improvement": improvement,
                "qlike_improvement": qlike_improvement,
                "passed": passed,
            }
        )

    improvements = np.asarray([row["rmse_improvement"] for row in target_results], dtype=float)
    pass_rate = float(np.mean([row["passed"] for row in target_results]))
    median_improvement = float(np.median(improvements))
    gate_passed = bool(pass_rate >= 2.0 / 3.0 and median_improvement >= 0.01)
    return {
        "milestone": "M5 daily SOTA-candidate model zoo",
        "claim": "calculus-native cross-asset jets beat stronger daily-data model-zoo baselines",
        "dataset": {
            "kind": "Yahoo daily OHLCV aligned close-to-close returns",
            "target_symbols": [symbol.upper() for symbol in target_symbols],
            "context_universe": [symbol.upper() for symbol in context_universe],
            "start": start,
            "end": end,
            "horizon": horizon,
        },
        "targets": target_results,
        "success_gate": {
            "target_pass_rate_required": 2.0 / 3.0,
            "target_pass_rate": pass_rate,
            "median_rmse_improvement_required": 0.01,
            "median_rmse_improvement": median_improvement,
            "passed": gate_passed,
        },
        "analysis": _sota_candidate_analysis(target_results=target_results, passed=gate_passed),
    }


def evaluate_milestone6(
    *,
    target_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    intraday_range: str = "60d",
    interval: str = "5m",
    horizon: int = 6,
    cache_dir: Path = Path("data/financial_signal_discovery"),
    refresh: bool = False,
) -> dict[str, object]:
    """Intraday 5-minute short-horizon volatility benchmark."""

    target_results = []
    for target_symbol in target_symbols:
        market = fetch_yahoo_intraday_ohlcv(
            symbol=target_symbol,
            intraday_range=intraday_range,
            interval=interval,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        dataset = intraday_volatility_dataset_from_ohlcv(market, horizon=horizon)
        timestamps = market.timestamp[1:]
        baseline_split = _chronological_split(
            *build_intraday_standard_features(dataset=dataset, timestamps=timestamps)
        )
        candidate_split = _chronological_split(
            *build_intraday_compact_calculus_features(dataset=dataset, timestamps=timestamps)
        )

        baseline_results = _evaluate_model_zoo(
            baseline_split,
            "intraday_standard",
            include_boosters=True,
        )
        candidate_results = _evaluate_model_zoo(
            candidate_split,
            "intraday_compact_roughness_jets",
            include_boosters=True,
        )
        best_baseline = _best_model_result(baseline_results)
        best_candidate = _best_model_result(candidate_results)
        improvement = (best_baseline["rmse"] - best_candidate["rmse"]) / best_baseline["rmse"]
        qlike_improvement = (best_baseline["qlike"] - best_candidate["qlike"]) / abs(
            best_baseline["qlike"]
        )
        passed = bool(improvement >= 0.01 and best_candidate["qlike"] <= best_baseline["qlike"])
        target_results.append(
            {
                "target_symbol": target_symbol.upper(),
                "best_baseline": best_baseline,
                "best_candidate": best_candidate,
                "baseline_rmse": best_baseline["rmse"],
                "candidate_rmse": best_candidate["rmse"],
                "baseline_qlike": best_baseline["qlike"],
                "candidate_qlike": best_candidate["qlike"],
                "rmse_improvement": improvement,
                "qlike_improvement": qlike_improvement,
                "passed": passed,
            }
        )

    improvements = np.asarray([row["rmse_improvement"] for row in target_results], dtype=float)
    pass_rate = float(np.mean([row["passed"] for row in target_results]))
    median_improvement = float(np.median(improvements))
    gate_passed = bool(pass_rate >= 2.0 / 3.0 and median_improvement >= 0.01)
    return {
        "milestone": "M6 intraday short-horizon volatility",
        "claim": "compact calculus jets improve 5-minute short-horizon volatility forecasting",
        "dataset": {
            "kind": "Yahoo intraday OHLCV",
            "target_symbols": [symbol.upper() for symbol in target_symbols],
            "range": intraday_range,
            "interval": interval,
            "horizon_bars": horizon,
            "target": "next-horizon annualized realized volatility",
        },
        "targets": target_results,
        "success_gate": {
            "target_pass_rate_required": 2.0 / 3.0,
            "target_pass_rate": pass_rate,
            "median_rmse_improvement_required": 0.01,
            "median_rmse_improvement": median_improvement,
            "passed": gate_passed,
        },
        "analysis": _intraday_analysis(target_results=target_results, passed=gate_passed),
    }


def evaluate_milestone7(
    *,
    assets: tuple[str, ...] = ("BTC", "ETH", "SOL"),
    date: str = "2026-03-06",
    horizon: int = 6,
    cache_dir: Path = Path("data/financial_signal_discovery"),
    max_samples_per_asset: int = 25_000,
) -> dict[str, object]:
    """Real order-book benchmark on Polymarket top-10 depth snapshots."""

    orderbook = fetch_polymarket_orderbook(date=date, cache_dir=cache_dir)
    target_results = []
    for asset in assets:
        standard = _chronological_split(
            *build_polymarket_orderbook_features(
                orderbook=orderbook,
                asset=asset,
                horizon=horizon,
                include_calculus=False,
                max_samples=max_samples_per_asset,
            )
        )
        calculus = _chronological_split(
            *build_polymarket_orderbook_features(
                orderbook=orderbook,
                asset=asset,
                horizon=horizon,
                include_calculus=True,
                max_samples=max_samples_per_asset,
            )
        )
        baseline_results = _evaluate_model_zoo(
            standard,
            "lob_standard_depth",
            include_boosters=True,
        )
        candidate_results = _evaluate_model_zoo(
            calculus,
            "lob_depth_plus_calculus_jets",
            include_boosters=True,
        )
        best_baseline = _best_model_result(baseline_results)
        best_candidate = _best_model_result(candidate_results)
        improvement = (best_baseline["rmse"] - best_candidate["rmse"]) / best_baseline["rmse"]
        qlike_improvement = (best_baseline["qlike"] - best_candidate["qlike"]) / abs(
            best_baseline["qlike"]
        )
        passed = bool(improvement >= 0.01 and best_candidate["qlike"] <= best_baseline["qlike"])
        target_results.append(
            {
                "asset": asset.upper(),
                "best_baseline": best_baseline,
                "best_candidate": best_candidate,
                "baseline_rmse": best_baseline["rmse"],
                "candidate_rmse": best_candidate["rmse"],
                "baseline_qlike": best_baseline["qlike"],
                "candidate_qlike": best_candidate["qlike"],
                "rmse_improvement": improvement,
                "qlike_improvement": qlike_improvement,
                "passed": passed,
            }
        )

    improvements = np.asarray([row["rmse_improvement"] for row in target_results], dtype=float)
    pass_rate = float(np.mean([row["passed"] for row in target_results]))
    median_improvement = float(np.median(improvements))
    gate_passed = bool(pass_rate >= 2.0 / 3.0 and median_improvement >= 0.01)
    return {
        "milestone": "M7 real order-book microstructure",
        "claim": "calculus jets over order-book states improve short-horizon mid-price movement forecasting",
        "dataset": {
            "kind": "Polymarket 10-level order-book snapshots from Hugging Face",
            "repo": "BrockMisner/polymarket-crypto-5m-15m",
            "date": date,
            "assets": [asset.upper() for asset in assets],
            "horizon_snapshots": horizon,
            "target": "future realized mid-price movement over horizon plus epsilon",
        },
        "targets": target_results,
        "success_gate": {
            "target_pass_rate_required": 2.0 / 3.0,
            "target_pass_rate": pass_rate,
            "median_rmse_improvement_required": 0.01,
            "median_rmse_improvement": median_improvement,
            "passed": gate_passed,
        },
        "analysis": _orderbook_analysis(target_results=target_results, passed=gate_passed),
    }


def evaluate_milestone8(
    *,
    train_file: str = "Train_Dst_NoAuction_DecPre_CF_7.txt",
    test_file: str = "Test_Dst_NoAuction_DecPre_CF_7.txt",
    label_horizon_index: int = 0,
    max_train_samples: int = 60_000,
    max_test_samples: int = 30_000,
    cache_dir: Path = Path("data/financial_signal_discovery"),
) -> dict[str, object]:
    """FI-2010/DeepLOB-style mid-price movement classification benchmark."""

    zip_path = fetch_deeplob_fi2010_zip(cache_dir=cache_dir)
    x_train_raw, y_train = load_fi2010_matrix(
        zip_path=zip_path,
        member=train_file,
        label_horizon_index=label_horizon_index,
        max_samples=max_train_samples,
    )
    x_test_raw, y_test = load_fi2010_matrix(
        zip_path=zip_path,
        member=test_file,
        label_horizon_index=label_horizon_index,
        max_samples=max_test_samples,
    )
    baseline_result = _evaluate_fi2010_model_zoo(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_test=x_test_raw,
        y_test=y_test,
        feature_set="fi2010_lob40",
    )
    x_train_calc = build_fi2010_calculus_features(x_train_raw)
    x_test_calc = build_fi2010_calculus_features(x_test_raw)
    candidate_result = _evaluate_fi2010_model_zoo(
        x_train_raw=x_train_calc,
        y_train=y_train,
        x_test=x_test_calc,
        y_test=y_test,
        feature_set="fi2010_lob40_plus_calculus_jets",
    )
    accuracy_improvement = candidate_result["accuracy"] - baseline_result["accuracy"]
    macro_f1_improvement = candidate_result["macro_f1"] - baseline_result["macro_f1"]
    gate_passed = bool(accuracy_improvement >= 0.01 and macro_f1_improvement >= 0.0)
    return {
        "milestone": "M8 FI-2010 DeepLOB-style classification",
        "claim": "calculus jets improve canonical FI-2010 mid-price movement classification over boosted LOB baselines",
        "dataset": {
            "kind": "FI-2010 decimal-precision no-auction subset from DeepLOB example data",
            "train_file": train_file,
            "test_file": test_file,
            "label_horizon_index": label_horizon_index,
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
            "target": "3-class mid-price movement label",
        },
        "models": {
            "best_baseline": baseline_result,
            "best_candidate": candidate_result,
            "candidate_family": [candidate_result],
        },
        "success_gate": {
            "accuracy_improvement_required": 0.01,
            "accuracy_improvement": accuracy_improvement,
            "macro_f1_not_worse": macro_f1_improvement >= 0.0,
            "macro_f1_improvement": macro_f1_improvement,
            "passed": gate_passed,
        },
        "analysis": _fi2010_analysis(
            baseline=baseline_result,
            candidate=candidate_result,
            passed=gate_passed,
        ),
    }


def evaluate_milestone9(
    *,
    train_file: str = "Train_Dst_NoAuction_DecPre_CF_7.txt",
    test_file: str = "Test_Dst_NoAuction_DecPre_CF_7.txt",
    label_horizon_index: int = 0,
    max_train_samples: int = 40_000,
    max_test_samples: int = 20_000,
    sequence_window: int = 100,
    epochs: int = 4,
    cache_dir: Path = Path("data/financial_signal_discovery"),
) -> dict[str, object]:
    """DeepLOB-style temporal neural benchmark with and without calculus channels."""

    zip_path = fetch_deeplob_fi2010_zip(cache_dir=cache_dir)
    x_train_raw, y_train = load_fi2010_matrix(
        zip_path=zip_path,
        member=train_file,
        label_horizon_index=label_horizon_index,
        max_samples=max_train_samples,
    )
    x_test_raw, y_test = load_fi2010_matrix(
        zip_path=zip_path,
        member=test_file,
        label_horizon_index=label_horizon_index,
        max_samples=max_test_samples,
    )
    baseline_result = _evaluate_fi2010_sequence_model(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_test=x_test_raw,
        y_test=y_test,
        feature_set="fi2010_sequence_lob40",
        sequence_window=sequence_window,
        epochs=epochs,
    )
    candidate_result = _evaluate_fi2010_sequence_model(
        x_train_raw=build_fi2010_calculus_features(x_train_raw),
        y_train=y_train,
        x_test=build_fi2010_calculus_features(x_test_raw),
        y_test=y_test,
        feature_set="fi2010_sequence_lob40_plus_calculus_jets",
        sequence_window=sequence_window,
        epochs=epochs,
    )
    accuracy_improvement = candidate_result["accuracy"] - baseline_result["accuracy"]
    macro_f1_improvement = candidate_result["macro_f1"] - baseline_result["macro_f1"]
    gate_passed = bool(accuracy_improvement >= 0.005 and macro_f1_improvement >= 0.0)
    return {
        "milestone": "M9 FI-2010 neural sequence calculus channels",
        "claim": "calculus channels improve a DeepLOB-style temporal neural baseline on FI-2010",
        "dataset": {
            "kind": "FI-2010 decimal-precision no-auction subset from DeepLOB example data",
            "train_file": train_file,
            "test_file": test_file,
            "label_horizon_index": label_horizon_index,
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
            "sequence_window": sequence_window,
            "epochs": epochs,
            "target": "3-class mid-price movement label",
        },
        "models": {
            "best_baseline": baseline_result,
            "best_candidate": candidate_result,
            "candidate_family": [candidate_result],
        },
        "success_gate": {
            "accuracy_improvement_required": 0.005,
            "accuracy_improvement": accuracy_improvement,
            "macro_f1_not_worse": macro_f1_improvement >= 0.0,
            "macro_f1_improvement": macro_f1_improvement,
            "passed": gate_passed,
        },
        "analysis": _fi2010_sequence_analysis(
            baseline=baseline_result,
            candidate=candidate_result,
            passed=gate_passed,
        ),
    }


def evaluate_milestone10(
    *,
    train_file: str = "Train_Dst_NoAuction_DecPre_CF_7.txt",
    test_file: str = "Test_Dst_NoAuction_DecPre_CF_7.txt",
    label_horizon_index: int = 0,
    max_train_samples: int = 40_000,
    max_test_samples: int = 20_000,
    sequence_window: int = 100,
    epochs: int = 4,
    cache_dir: Path = Path("data/financial_signal_discovery"),
) -> dict[str, object]:
    """DeepLOB/Inception-LSTM-style ablation with calculus channels."""

    zip_path = fetch_deeplob_fi2010_zip(cache_dir=cache_dir)
    x_train_raw, y_train = load_fi2010_matrix(
        zip_path=zip_path,
        member=train_file,
        label_horizon_index=label_horizon_index,
        max_samples=max_train_samples,
    )
    x_test_raw, y_test = load_fi2010_matrix(
        zip_path=zip_path,
        member=test_file,
        label_horizon_index=label_horizon_index,
        max_samples=max_test_samples,
    )
    baseline_result = _evaluate_fi2010_sequence_model(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_test=x_test_raw,
        y_test=y_test,
        feature_set="fi2010_deeplob_lob40",
        sequence_window=sequence_window,
        epochs=epochs,
        architecture="inception_lstm",
    )
    candidate_results = [
        _evaluate_fi2010_sequence_model(
            x_train_raw=build_fi2010_calculus_features(x_train_raw),
            y_train=y_train,
            x_test=build_fi2010_calculus_features(x_test_raw),
            y_test=y_test,
            feature_set="fi2010_deeplob_lob40_plus_calculus_jets",
            sequence_window=sequence_window,
            epochs=epochs,
            architecture="inception_lstm",
        ),
        _evaluate_fi2010_sequence_model(
            x_train_raw=build_fi2010_compact_calculus_features(x_train_raw),
            y_train=y_train,
            x_test=build_fi2010_compact_calculus_features(x_test_raw),
            y_test=y_test,
            feature_set="fi2010_deeplob_lob40_plus_roughness_jets",
            sequence_window=sequence_window,
            epochs=epochs,
            architecture="inception_lstm",
        ),
        _evaluate_fi2010_sequence_model(
            x_train_raw=build_fi2010_compact_calculus_features(x_train_raw),
            y_train=y_train,
            x_test=build_fi2010_compact_calculus_features(x_test_raw),
            y_test=y_test,
            feature_set="fi2010_deeplob_lob40_plus_roughness_fusion",
            sequence_window=sequence_window,
            epochs=epochs,
            architecture="inception_lstm_calculus_fusion",
        ),
    ]
    candidate_result = max(
        candidate_results,
        key=lambda result: (result["validation_accuracy"], result["validation_macro_f1"]),
    )
    accuracy_improvement = candidate_result["accuracy"] - baseline_result["accuracy"]
    macro_f1_improvement = candidate_result["macro_f1"] - baseline_result["macro_f1"]
    gate_passed = bool(accuracy_improvement >= 0.005 and macro_f1_improvement >= 0.0)
    return {
        "milestone": "M10 FI-2010 DeepLOB Inception-LSTM calculus channels",
        "claim": "calculus channels improve a DeepLOB/Inception-LSTM-style temporal baseline on FI-2010",
        "dataset": {
            "kind": "FI-2010 decimal-precision no-auction subset from DeepLOB example data",
            "train_file": train_file,
            "test_file": test_file,
            "label_horizon_index": label_horizon_index,
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
            "sequence_window": sequence_window,
            "epochs": epochs,
            "target": "3-class mid-price movement label",
        },
        "models": {
            "best_baseline": baseline_result,
            "best_candidate": candidate_result,
            "candidate_family": candidate_results,
        },
        "success_gate": {
            "accuracy_improvement_required": 0.005,
            "accuracy_improvement": accuracy_improvement,
            "macro_f1_not_worse": macro_f1_improvement >= 0.0,
            "macro_f1_improvement": macro_f1_improvement,
            "passed": gate_passed,
        },
        "analysis": _fi2010_deeplob_analysis(
            baseline=baseline_result,
            candidate=candidate_result,
            passed=gate_passed,
        ),
    }


def volatility_dataset_from_returns(
    returns: np.ndarray,
    *,
    horizon: int = 5,
    realized_window: int = 5,
    feature_start: int = 90,
) -> VolatilityDataset:
    returns = np.asarray(returns, dtype=float)
    observed_vol = _rolling_rms(returns, window=realized_window) * np.sqrt(252.0)
    target_vol = np.asarray(
        [
            np.sqrt(np.mean(returns[t + 1 : t + horizon + 1] ** 2)) * np.sqrt(252.0)
            if t + horizon < returns.size
            else np.nan
            for t in range(returns.size)
        ],
        dtype=float,
    )
    return VolatilityDataset(
        returns=returns,
        latent_vol=observed_vol,
        observed_vol=observed_vol,
        target_vol=target_vol,
        feature_start=feature_start,
        horizon=horizon,
    )


def fetch_yahoo_intraday_ohlcv(
    *,
    symbol: str = "SPY",
    intraday_range: str = "60d",
    interval: str = "5m",
    cache_dir: Path = Path("data/financial_signal_discovery"),
    refresh: bool = False,
    timeout: int = 30,
) -> MarketOHLCV:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol.upper()}_{intraday_range}_{interval}_yahoo_intraday.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text())
    else:
        url = _yahoo_intraday_url(symbol=symbol, intraday_range=intraday_range, interval=interval)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cache_path.write_text(json.dumps(payload))
    return _parse_yahoo_chart(symbol=symbol, payload=payload)


def fetch_polymarket_orderbook(
    *,
    date: str = "2026-03-06",
    cache_dir: Path = Path("data/financial_signal_discovery"),
    repo_id: str = "BrockMisner/polymarket-crypto-5m-15m",
) -> object:
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover
        raise ImportError("M7 requires pandas and huggingface_hub") from exc

    local_dir = cache_dir / "polymarket_orderbooks"
    local_dir.mkdir(parents=True, exist_ok=True)
    filename = f"orderbooks/{date}.parquet"
    source = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=local_dir,
    )
    return pd.read_parquet(source)


def fetch_deeplob_fi2010_zip(
    *,
    cache_dir: Path = Path("data/financial_signal_discovery"),
    timeout: int = 180,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "deeplob_fi2010_data.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    url = (
        "https://raw.githubusercontent.com/zcakhaa/"
        "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
        "master/data/data.zip"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as response:
        zip_path.write_bytes(response.read())
    return zip_path


def align_market_returns(markets: list[MarketOHLCV]) -> np.ndarray:
    return _align_market_series(
        markets,
        series=[np.diff(np.log(np.maximum(market.adj_close, 1e-12))) for market in markets],
        timestamps=[_timestamp_dates(market.timestamp[1:]) for market in markets],
    )


def align_market_adj_close(markets: list[MarketOHLCV]) -> np.ndarray:
    return _align_market_series(
        markets,
        series=[market.adj_close[1:] for market in markets],
        timestamps=[_timestamp_dates(market.timestamp[1:]) for market in markets],
    )


def _align_market_series(
    markets: list[MarketOHLCV],
    *,
    series: list[np.ndarray],
    timestamps: list[np.ndarray],
) -> np.ndarray:
    if not markets:
        raise ValueError("At least one market is required")
    timestamp_sets = [set(stamps.tolist()) for stamps in timestamps]
    common = sorted(set.intersection(*timestamp_sets))
    if len(common) < 300:
        raise ValueError("Too few overlapping market observations")
    aligned = []
    for stamps, values in zip(timestamps, series, strict=True):
        by_time = dict(zip(stamps.tolist(), values, strict=True))
        aligned.append([by_time[timestamp] for timestamp in common])
    return np.asarray(aligned, dtype=float).T


def build_cross_asset_features(
    *,
    returns: np.ndarray,
    levels: np.ndarray | None = None,
    symbols: tuple[str, ...],
    target_index: int = 0,
    horizon: int = 5,
    lookbacks: tuple[int, ...] = (11, 22, 63),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_dataset = volatility_dataset_from_returns(
        returns[:, target_index],
        horizon=horizon,
        feature_start=max(90, max(lookbacks) + 5),
    )
    own_x, targets, own_names = build_multiscale_jet_features(target_dataset, lookbacks=lookbacks)
    vols = np.stack([_rolling_rms(returns[:, idx], window=5) * np.sqrt(252.0) for idx in range(returns.shape[1])])
    rows = []
    for t in _valid_feature_times(target_dataset):
        row = []
        for asset_idx, _symbol in enumerate(symbols):
            if asset_idx == target_index:
                continue
            vol = vols[asset_idx]
            row.extend([vol[t], float(np.mean(vol[t - 4 : t + 1])), float(np.mean(vol[t - 21 : t + 1]))])
            log_vol = np.log(np.maximum(vol, 1e-8))
            for lookback in lookbacks:
                jets = _local_taylor_jets(log_vol[t - lookback + 1 : t + 1])
                row.extend(jets[[1, 2, 4]].tolist())
            if levels is not None:
                log_level = np.log(np.maximum(levels[:, asset_idx], 1e-8))
                row.extend(
                    [
                        log_level[t],
                        float(np.mean(log_level[t - 4 : t + 1])),
                        float(np.mean(log_level[t - 21 : t + 1])),
                    ]
                )
                for lookback in lookbacks:
                    jets = _local_taylor_jets(log_level[t - lookback + 1 : t + 1])
                    row.extend(jets[[1, 2, 4]].tolist())
        rows.append(row)
    cross_names = []
    for asset_idx, symbol in enumerate(symbols):
        if asset_idx == target_index:
            continue
        cross_names.extend([f"{symbol}_rv_1", f"{symbol}_rv_5", f"{symbol}_rv_22"])
        for lookback in lookbacks:
            cross_names.extend(
                [
                    f"{symbol}_log_rv_{lookback}_slope",
                    f"{symbol}_log_rv_{lookback}_curvature",
                    f"{symbol}_log_rv_{lookback}_roughness",
                ]
            )
        if levels is not None:
            cross_names.extend([f"{symbol}_log_level_1", f"{symbol}_log_level_5", f"{symbol}_log_level_22"])
            for lookback in lookbacks:
                cross_names.extend(
                    [
                        f"{symbol}_log_level_{lookback}_slope",
                        f"{symbol}_log_level_{lookback}_curvature",
                        f"{symbol}_log_level_{lookback}_roughness",
                    ]
                )
    return np.concatenate([own_x, np.asarray(rows, dtype=float)], axis=1), targets, [*own_names, *cross_names]


def build_cross_asset_standard_features(
    *,
    returns: np.ndarray,
    levels: np.ndarray,
    symbols: tuple[str, ...],
    target_index: int = 0,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_dataset = volatility_dataset_from_returns(
        returns[:, target_index],
        horizon=horizon,
        feature_start=90,
    )
    own_x, targets, own_names = build_har_features(target_dataset)
    vols = np.stack([_rolling_rms(returns[:, idx], window=5) * np.sqrt(252.0) for idx in range(returns.shape[1])])
    rows = []
    for t in _valid_feature_times(target_dataset):
        row = []
        for asset_idx, _symbol in enumerate(symbols):
            vol = vols[asset_idx]
            log_level = np.log(np.maximum(levels[:, asset_idx], 1e-8))
            if asset_idx != target_index:
                row.extend(
                    [
                        vol[t],
                        float(np.mean(vol[t - 4 : t + 1])),
                        float(np.mean(vol[t - 21 : t + 1])),
                    ]
                )
            row.extend(
                [
                    returns[t, asset_idx],
                    float(np.mean(returns[t - 4 : t + 1, asset_idx])),
                    float(np.mean(returns[t - 21 : t + 1, asset_idx])),
                    log_level[t],
                    float(np.mean(log_level[t - 4 : t + 1])),
                    float(np.mean(log_level[t - 21 : t + 1])),
                ]
            )
        rows.append(row)

    names = []
    for asset_idx, symbol in enumerate(symbols):
        if asset_idx != target_index:
            names.extend([f"{symbol}_rv_1", f"{symbol}_rv_5", f"{symbol}_rv_22"])
        names.extend(
            [
                f"{symbol}_ret_1",
                f"{symbol}_ret_5",
                f"{symbol}_ret_22",
                f"{symbol}_log_level_1",
                f"{symbol}_log_level_5",
                f"{symbol}_log_level_22",
            ]
        )
    return np.concatenate([own_x, np.asarray(rows, dtype=float)], axis=1), targets, [*own_names, *names]


def build_cross_asset_sota_candidate_features(
    *,
    returns: np.ndarray,
    levels: np.ndarray,
    symbols: tuple[str, ...],
    target_index: int = 0,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    standard_x, targets, standard_names = build_cross_asset_standard_features(
        returns=returns,
        levels=levels,
        symbols=symbols,
        target_index=target_index,
        horizon=horizon,
    )
    jet_x, jet_targets, jet_names = build_cross_asset_features(
        returns=returns,
        levels=levels,
        symbols=symbols,
        target_index=target_index,
        horizon=horizon,
    )
    if not np.allclose(targets, jet_targets):
        raise ValueError("Standard and jet feature targets are not aligned")
    return (
        np.concatenate([standard_x, jet_x], axis=1),
        targets,
        [*standard_names, *[f"jet__{name}" for name in jet_names]],
    )


def build_cross_asset_compact_calculus_features(
    *,
    returns: np.ndarray,
    levels: np.ndarray,
    symbols: tuple[str, ...],
    target_index: int = 0,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    standard_x, targets, standard_names = build_cross_asset_standard_features(
        returns=returns,
        levels=levels,
        symbols=symbols,
        target_index=target_index,
        horizon=horizon,
    )
    jet_x, jet_targets, jet_names = build_cross_asset_features(
        returns=returns,
        levels=levels,
        symbols=symbols,
        target_index=target_index,
        horizon=horizon,
    )
    if not np.allclose(targets, jet_targets):
        raise ValueError("Standard and compact jet feature targets are not aligned")
    keep = np.asarray(["roughness" in name for name in jet_names], dtype=bool)
    compact = jet_x[:, keep]
    compact_names = [f"jet__{name}" for name, selected in zip(jet_names, keep, strict=True) if selected]
    return np.concatenate([standard_x, compact], axis=1), targets, [*standard_names, *compact_names]


def intraday_volatility_dataset_from_ohlcv(
    ohlcv: MarketOHLCV,
    *,
    horizon: int = 6,
    realized_window: int = 6,
    bars_per_year: float = 252.0 * 78.0,
) -> VolatilityDataset:
    close = np.maximum(ohlcv.close, 1e-12)
    returns = np.diff(np.log(close))
    scale = np.sqrt(bars_per_year)
    observed_vol = _rolling_rms(returns, window=realized_window) * scale
    target_vol = np.asarray(
        [
            np.sqrt(np.mean(returns[t + 1 : t + horizon + 1] ** 2)) * scale
            if t + horizon < returns.size
            else np.nan
            for t in range(returns.size)
        ],
        dtype=float,
    )
    return VolatilityDataset(
        returns=returns,
        latent_vol=observed_vol,
        observed_vol=observed_vol,
        target_vol=target_vol,
        feature_start=390,
        horizon=horizon,
    )


def build_intraday_standard_features(
    *,
    dataset: VolatilityDataset,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if timestamps.shape[0] != dataset.returns.shape[0]:
        raise ValueError("Intraday timestamps must align with return observations")
    vol = dataset.observed_vol
    returns = dataset.returns
    time_features, time_names = _intraday_time_features(timestamps)
    rows = []
    targets = []
    for t in _valid_feature_times(dataset):
        rows.append(
            [
                vol[t],
                float(np.mean(vol[t - 5 : t + 1])),
                float(np.mean(vol[t - 77 : t + 1])),
                float(np.mean(vol[t - 389 : t + 1])),
                abs(returns[t]),
                float(np.mean(np.abs(returns[t - 5 : t + 1]))),
                float(np.mean(np.abs(returns[t - 77 : t + 1]))),
                returns[t],
                float(np.mean(returns[t - 5 : t + 1])),
                float(np.mean(returns[t - 77 : t + 1])),
                *time_features[t].tolist(),
            ]
        )
        targets.append(dataset.target_vol[t])
    names = [
        "rv_1",
        "rv_6",
        "rv_78",
        "rv_390",
        "abs_ret_1",
        "abs_ret_6",
        "abs_ret_78",
        "ret_1",
        "ret_6",
        "ret_78",
        *time_names,
    ]
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), names


def build_intraday_compact_calculus_features(
    *,
    dataset: VolatilityDataset,
    timestamps: np.ndarray,
    lookbacks: tuple[int, ...] = (12, 78, 390),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    standard_x, targets, standard_names = build_intraday_standard_features(
        dataset=dataset,
        timestamps=timestamps,
    )
    vol = np.log(np.maximum(dataset.observed_vol, 1e-8))
    rows = []
    for t in _valid_feature_times(dataset):
        row = []
        for lookback in lookbacks:
            jets = _local_taylor_jets(vol[t - lookback + 1 : t + 1])
            row.extend([jets[1], jets[2], jets[4]])
        rows.append(row)
    jet_names = [
        f"jet_log_rv_{lookback}_{name}"
        for lookback in lookbacks
        for name in ("slope", "curvature", "roughness")
    ]
    return (
        np.concatenate([standard_x, np.asarray(rows, dtype=float)], axis=1),
        targets,
        [*standard_names, *jet_names],
    )


def build_polymarket_orderbook_features(
    *,
    orderbook: object,
    asset: str,
    horizon: int = 6,
    include_calculus: bool = False,
    max_samples: int = 25_000,
    min_group_size: int = 36,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = orderbook
    asset_df = df[df["asset"].astype(str).str.upper() == asset.upper()].copy()
    if asset_df.empty:
        raise ValueError(f"No order-book rows for asset {asset}")
    asset_df = asset_df.sort_values(["timestamp", "market_id", "token_id"])
    rows: list[list[float]] = []
    targets: list[float] = []
    names = _polymarket_standard_feature_names()
    if include_calculus:
        names = [*names, *_polymarket_calculus_feature_names()]

    grouped = asset_df.groupby("token_id", sort=False)
    for _, group in grouped:
        if len(group) < min_group_size:
            continue
        group = group.sort_values("timestamp")
        raw_mids = group["mid_price"].to_numpy(dtype=float)
        raw_spreads = group["spread"].to_numpy(dtype=float)
        raw_bid_levels = [_parse_book_levels(value) for value in group["bid_levels"].tolist()]
        raw_ask_levels = [_parse_book_levels(value) for value in group["ask_levels"].tolist()]
        valid = [
            idx
            for idx, (bid, ask) in enumerate(zip(raw_bid_levels, raw_ask_levels, strict=True))
            if bid and ask and np.isfinite(raw_mids[idx]) and np.isfinite(raw_spreads[idx])
        ]
        if len(valid) < min_group_size:
            continue
        mids = raw_mids[valid]
        spreads = raw_spreads[valid]
        bid_levels = [raw_bid_levels[idx] for idx in valid]
        ask_levels = [raw_ask_levels[idx] for idx in valid]
        standard_matrix = np.asarray(
            [
                _polymarket_standard_row(
                    mid=mids[idx],
                    spread=spreads[idx],
                    bid_levels=bid_levels[idx],
                    ask_levels=ask_levels[idx],
                )
                for idx in range(len(mids))
            ],
            dtype=float,
        )
        imbalance = standard_matrix[:, 9]
        for idx in range(20, len(mids) - horizon):
            row = standard_matrix[idx].tolist()
            if include_calculus:
                row.extend(_polymarket_calculus_row(mids, spreads, imbalance, idx))
            future_steps = np.diff(mids[idx : idx + horizon + 1])
            target = float(np.sqrt(np.mean(future_steps * future_steps)) + 1e-4)
            rows.append(row)
            targets.append(float(target))
            if len(rows) >= max_samples:
                return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), names

    if len(rows) < 300:
        raise ValueError(f"Too few Polymarket order-book samples for {asset}")
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), names


def load_fi2010_matrix(
    *,
    zip_path: Path,
    member: str,
    label_horizon_index: int = 0,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as file:
            matrix = np.loadtxt(file)
    if matrix.shape[0] < 149:
        raise ValueError("FI-2010 matrix must contain 149 rows")
    if max_samples is not None:
        matrix = matrix[:, :max_samples]
    x = matrix[:40].T.astype(float)
    label_row = 144 + label_horizon_index
    y = matrix[label_row].astype(int) - 1
    if not np.all((0 <= y) & (y <= 2)):
        raise ValueError("FI-2010 labels must map to classes 0, 1, 2")
    return x, y


def build_fi2010_calculus_features(x: np.ndarray, *, lookback: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mid = 0.5 * (x[:, 0] + x[:, 2])
    spread = x[:, 0] - x[:, 2]
    imbalance = (x[:, 3] - x[:, 1]) / (x[:, 3] + x[:, 1] + 1e-12)
    rows = []
    for idx in range(x.shape[0]):
        if idx < lookback:
            rows.append([0.0] * 9)
            continue
        row = []
        for series in (mid, spread, imbalance):
            jets = _local_taylor_jets(series[idx - lookback + 1 : idx + 1])
            row.extend([float(jets[1]), float(jets[2]), float(jets[4])])
        rows.append(row)
    return np.concatenate([x, np.asarray(rows, dtype=float)], axis=1)


def build_fi2010_compact_calculus_features(x: np.ndarray, *, lookback: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mid = 0.5 * (x[:, 0] + x[:, 2])
    spread = x[:, 0] - x[:, 2]
    imbalance = (x[:, 3] - x[:, 1]) / (x[:, 3] + x[:, 1] + 1e-12)
    rows = []
    for idx in range(x.shape[0]):
        if idx < lookback:
            rows.append([0.0] * 3)
            continue
        row = []
        for series in (mid, spread, imbalance):
            jets = _local_taylor_jets(series[idx - lookback + 1 : idx + 1])
            row.append(float(jets[4]))
        rows.append(row)
    return np.concatenate([x, np.asarray(rows, dtype=float)], axis=1)


def make_sequence_windows(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sequence_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if sequence_window <= 1:
        raise ValueError("sequence_window must be greater than 1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Features and labels must have matching lengths")
    if x.shape[0] < sequence_window:
        raise ValueError("Not enough samples for requested sequence window")
    windows = np.stack(
        [x[idx - sequence_window + 1 : idx + 1] for idx in range(sequence_window - 1, x.shape[0])],
        axis=0,
    )
    return windows.astype(np.float32), y[sequence_window - 1 :].astype(np.int64)


def write_artifacts(results: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    if results.get("milestone") in {
        "M8 FI-2010 DeepLOB-style classification",
        "M9 FI-2010 neural sequence calculus channels",
        "M10 FI-2010 DeepLOB Inception-LSTM calculus channels",
    }:
        _write_classification_report(results, out_dir)
        return
    if "targets" in results:
        _write_robustness_report(results, out_dir)
        return
    models = results["models"]
    gate = results["success_gate"]
    assert isinstance(models, dict)
    assert isinstance(gate, dict)
    model_items = list(models.items())
    baseline_name, baseline = model_items[0]
    candidate_name, candidate = model_items[1]
    lines = [
        f"# Financial Signal Discovery {results['milestone']}",
        "",
        str(results["claim"]),
        "",
        "## Results",
        "",
        f"- {baseline_name} RMSE: `{baseline['rmse']:.6f}`, QLIKE `{baseline['qlike']:.6f}`",
        f"- {candidate_name} RMSE: `{candidate['rmse']:.6f}`, QLIKE `{candidate['qlike']:.6f}`",
        f"- RMSE improvement: `{gate['rmse_improvement']:.2%}`",
        f"- Success gate passed: `{gate['passed']}`",
        "",
        "## Selected Features",
        "",
    ]
    for row in candidate["selected_features"]:
        lines.append(f"- `{row['name']}` coefficient `{row['coefficient']:.6g}`")
    if "analysis" in results:
        lines.extend(["", "## Analysis", ""])
        for item in results["analysis"]:
            lines.append(f"- {item}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _write_classification_report(results: dict[str, object], out_dir: Path) -> None:
    gate = results["success_gate"]
    models = results["models"]
    baseline = models["best_baseline"]
    candidate = models["best_candidate"]
    lines = [
        f"# Financial Signal Discovery {results['milestone']}",
        "",
        str(results["claim"]),
        "",
        "## Results",
        "",
        f"- Baseline accuracy: `{baseline['accuracy']:.4f}`, macro-F1 `{baseline['macro_f1']:.4f}`",
        f"- Candidate accuracy: `{candidate['accuracy']:.4f}`, macro-F1 `{candidate['macro_f1']:.4f}`",
        f"- Accuracy improvement: `{gate['accuracy_improvement']:.2%}`",
        f"- Macro-F1 improvement: `{gate['macro_f1_improvement']:.4f}`",
        f"- Success gate passed: `{gate['passed']}`",
        "",
        "## Analysis",
        "",
    ]
    for item in results["analysis"]:
        lines.append(f"- {item}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _write_robustness_report(results: dict[str, object], out_dir: Path) -> None:
    gate = results["success_gate"]
    lines = [
        f"# Financial Signal Discovery {results['milestone']}",
        "",
        str(results["claim"]),
        "",
        "## Results",
        "",
        f"- Target pass rate: `{gate['target_pass_rate']:.2%}`",
        f"- Median RMSE improvement: `{gate['median_rmse_improvement']:.2%}`",
        f"- Success gate passed: `{gate['passed']}`",
        "",
        "## Targets",
        "",
    ]
    for row in results["targets"]:
        label = row.get("target_symbol", row.get("asset", "unknown"))
        lines.append(
            f"- `{label}` RMSE improvement `{row['rmse_improvement']:.2%}`, "
            f"QLIKE improvement `{row['qlike_improvement']:.2%}`, passed `{row['passed']}`"
        )
    if "analysis" in results:
        lines.extend(["", "## Analysis", ""])
        for item in results["analysis"]:
            lines.append(f"- {item}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _valid_feature_times(dataset: VolatilityDataset) -> range:
    stop = dataset.returns.size - dataset.horizon
    return range(dataset.feature_start, stop)


def _rolling_rms(x: np.ndarray, *, window: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for idx in range(x.size):
        left = max(0, idx - window + 1)
        out[idx] = np.sqrt(np.mean(x[left : idx + 1] ** 2))
    return out


def _local_taylor_jets(window: np.ndarray) -> np.ndarray:
    n = window.size
    x = np.linspace(-1.0, 0.0, n)
    design = np.stack([np.ones_like(x), x, x * x, x * x * x], axis=1)
    coef = np.linalg.lstsq(design, window, rcond=None)[0]
    fit = design @ coef
    return np.asarray([coef[0], coef[1], 2.0 * coef[2], 6.0 * coef[3], np.std(window - fit)], dtype=float)


def _chronological_split(x: np.ndarray, y: np.ndarray, names: list[str]) -> ChronologicalSplit:
    n = y.size
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    return ChronologicalSplit(
        x_train=x[:n_train],
        y_train=y[:n_train],
        x_val=x[n_train : n_train + n_val],
        y_val=y[n_train : n_train + n_val],
        x_test=x[n_train + n_val :],
        y_test=y[n_train + n_val :],
        names=names,
    )


def _fit_tuned_linear(split: ChronologicalSplit) -> FittedLinearModel:
    x_mean = split.x_train.mean(axis=0)
    x_scale = np.where(split.x_train.std(axis=0) < 1e-12, 1.0, split.x_train.std(axis=0))
    target_transform = "log"
    y_train_fit = np.log(np.maximum(split.y_train, 1e-8))
    y_mean = float(y_train_fit.mean())
    xtr = (split.x_train - x_mean) / x_scale
    xv = (split.x_val - x_mean) / x_scale
    best: tuple[float, float, np.ndarray, float] | None = None
    for alpha in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1_000.0, 10_000.0):
        coef = _ridge_coef(xtr, y_train_fit - y_mean, alpha)
        pred = np.exp(y_mean + xv @ coef)
        scale = _qlike_calibration_scale(split.y_val, pred)
        pred = scale * pred
        score = _forecast_metrics(split.y_val, pred)["rmse"]
        if best is None or score < best[0]:
            best = (score, alpha, coef, scale)
    assert best is not None
    _, alpha, coef, scale = best
    return FittedLinearModel(
        feature_names=split.names,
        coefficients=coef,
        intercept=y_mean,
        alpha=alpha,
        x_mean=x_mean,
        x_scale=x_scale,
        target_transform=target_transform,
        prediction_scale=scale,
    )


def _evaluate_linear_only(split: ChronologicalSplit, feature_set: str) -> list[dict[str, object]]:
    model = _fit_tuned_linear(split)
    val_pred = model.predict(split.x_val)
    test_pred = model.predict(split.x_test)
    val_metrics = _forecast_metrics(split.y_val, val_pred)
    test_metrics = _forecast_metrics(split.y_test, test_pred)
    return [
        _model_result(
            feature_set=feature_set,
            model_name="ridge_logvol",
            feature_count=len(split.names),
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            details={"alpha": model.alpha},
        )
    ]


def _evaluate_model_zoo(
    split: ChronologicalSplit,
    feature_set: str,
    *,
    include_boosters: bool,
) -> list[dict[str, object]]:
    results = _evaluate_linear_only(split, feature_set)
    if not include_boosters:
        return results
    for model_name, model in _tree_model_specs():
        results.append(_evaluate_log_target_regressor(split, feature_set, model_name, model))
    return results


def _evaluate_log_target_regressor(
    split: ChronologicalSplit,
    feature_set: str,
    model_name: str,
    model: object,
) -> dict[str, object]:
    y_train = np.log(np.maximum(split.y_train, 1e-8))
    model.fit(split.x_train, y_train)
    val_raw = np.exp(np.asarray(model.predict(split.x_val), dtype=float))
    scale = _qlike_calibration_scale(split.y_val, val_raw)
    val_pred = scale * val_raw
    test_pred = scale * np.exp(np.asarray(model.predict(split.x_test), dtype=float))
    return _model_result(
        feature_set=feature_set,
        model_name=model_name,
        feature_count=len(split.names),
        validation_metrics=_forecast_metrics(split.y_val, val_pred),
        test_metrics=_forecast_metrics(split.y_test, test_pred),
        details={"calibration_scale": scale},
    )


def _evaluate_fi2010_model_zoo(
    *,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_set: str,
) -> dict[str, object]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score

    n_train = int(0.8 * y_train.size)
    x_train = x_train_raw[:n_train]
    y_fit = y_train[:n_train]
    x_val = x_train_raw[n_train:]
    y_val = y_train[n_train:]
    specs: list[tuple[str, object]] = [
        (
            "sklearn_hgb_classifier",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=0,
            ),
        )
    ]
    try:
        from xgboost import XGBClassifier

        specs.append(
            (
                "xgboost_classifier_depth3",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    objective="multi:softprob",
                    num_class=3,
                    n_jobs=1,
                    random_state=0,
                    verbosity=0,
                ),
            )
        )
    except Exception:
        pass
    best: tuple[float, str, object, float] | None = None
    for model_name, model in specs:
        model.fit(x_train, y_fit)
        val_pred = model.predict(x_val)
        val_acc = float(accuracy_score(y_val, val_pred))
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        if best is None or val_acc > best[0]:
            best = (val_acc, model_name, model, val_f1)
    assert best is not None
    val_acc, model_name, model, val_f1 = best
    test_pred = model.predict(x_test)
    return {
        "feature_set": feature_set,
        "model_name": model_name,
        "feature_count": int(x_train_raw.shape[1]),
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro")),
    }


def _evaluate_fi2010_sequence_model(
    *,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_set: str,
    sequence_window: int,
    epochs: int,
    architecture: str = "cnn_gru",
) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, f1_score

    x_seq, y_seq = make_sequence_windows(x_train_raw, y_train, sequence_window=sequence_window)
    x_test_seq, y_test_seq = make_sequence_windows(x_test, y_test, sequence_window=sequence_window)
    n_train = int(0.8 * y_seq.size)
    x_fit = x_seq[:n_train]
    y_fit = y_seq[:n_train]
    x_val = x_seq[n_train:]
    y_val = y_seq[n_train:]
    best: tuple[float, float, object, np.ndarray, str] | None = None
    for loss_mode in ("unweighted", "class_weighted"):
        model, val_pred = _train_torch_sequence_classifier(
            x_train=x_fit,
            y_train=y_fit,
            x_val=x_val,
            epochs=epochs,
            class_weighted=loss_mode == "class_weighted",
            architecture=architecture,
        )
        val_acc = float(accuracy_score(y_val, val_pred))
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        if best is None or (val_acc, val_f1) > (best[0], best[1]):
            best = (val_acc, val_f1, model, val_pred, loss_mode)
    assert best is not None
    val_acc, val_f1, model, _val_pred, loss_mode = best
    test_pred = _predict_torch_sequence_classifier(model, x_test_seq)
    return {
        "feature_set": feature_set,
        "model_name": f"{architecture}_sequence_classifier",
        "feature_count": int(x_train_raw.shape[1]),
        "sequence_window": sequence_window,
        "epochs": epochs,
        "architecture": architecture,
        "loss_mode": loss_mode,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
        "accuracy": float(accuracy_score(y_test_seq, test_pred)),
        "macro_f1": float(f1_score(y_test_seq, test_pred, average="macro")),
    }


def _train_torch_sequence_classifier(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    epochs: int,
    class_weighted: bool = True,
    architecture: str = "cnn_gru",
) -> tuple[object, np.ndarray]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_scale = np.where(x_train.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, x_train.std(axis=(0, 1), keepdims=True))

    class CnnGruClassifier(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.gru = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
            self.head = nn.Linear(32, 3)
            self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
            self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = (x - self.x_mean) / self.x_scale
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            _, hidden = self.gru(x)
            return self.head(hidden[-1])

    class InceptionLstmClassifier(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.input_projection = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=1),
                nn.ReLU(),
            )
            self.branch_1 = nn.Conv1d(32, 24, kernel_size=1)
            self.branch_3 = nn.Conv1d(32, 24, kernel_size=3, padding=1)
            self.branch_5 = nn.Conv1d(32, 24, kernel_size=5, padding=2)
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(32, 24, kernel_size=1),
            )
            self.activation = nn.ReLU()
            self.lstm = nn.LSTM(input_size=96, hidden_size=48, batch_first=True)
            self.dropout = nn.Dropout(0.1)
            self.head = nn.Linear(48, 3)
            self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
            self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = (x - self.x_mean) / self.x_scale
            x = self.input_projection(x.transpose(1, 2))
            x = torch.cat(
                [
                    self.branch_1(x),
                    self.branch_3(x),
                    self.branch_5(x),
                    self.pool_branch(x),
                ],
                dim=1,
            )
            x = self.activation(x).transpose(1, 2)
            _, (hidden, _) = self.lstm(x)
            return self.head(self.dropout(hidden[-1]))

    class InceptionLstmCalculusFusionClassifier(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            if n_features <= 40:
                raise ValueError("Calculus fusion architecture requires extra calculus channels")
            self.raw_features = 40
            self.input_projection = nn.Sequential(
                nn.Conv1d(self.raw_features, 32, kernel_size=1),
                nn.ReLU(),
            )
            self.branch_1 = nn.Conv1d(32, 24, kernel_size=1)
            self.branch_3 = nn.Conv1d(32, 24, kernel_size=3, padding=1)
            self.branch_5 = nn.Conv1d(32, 24, kernel_size=5, padding=2)
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(32, 24, kernel_size=1),
            )
            self.activation = nn.ReLU()
            self.raw_lstm = nn.LSTM(input_size=96, hidden_size=48, batch_first=True)
            self.calc_gru = nn.GRU(input_size=n_features - self.raw_features, hidden_size=8, batch_first=True)
            self.dropout = nn.Dropout(0.1)
            self.head = nn.Linear(56, 3)
            self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
            self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = (x - self.x_mean) / self.x_scale
            raw = x[:, :, : self.raw_features]
            calc = x[:, :, self.raw_features :]
            raw = self.input_projection(raw.transpose(1, 2))
            raw = torch.cat(
                [
                    self.branch_1(raw),
                    self.branch_3(raw),
                    self.branch_5(raw),
                    self.pool_branch(raw),
                ],
                dim=1,
            )
            raw = self.activation(raw).transpose(1, 2)
            _, (raw_hidden, _) = self.raw_lstm(raw)
            _, calc_hidden = self.calc_gru(calc)
            fused = torch.cat([raw_hidden[-1], calc_hidden[-1]], dim=1)
            return self.head(self.dropout(fused))

    if architecture == "cnn_gru":
        model = CnnGruClassifier(x_train.shape[2])
    elif architecture == "inception_lstm":
        model = InceptionLstmClassifier(x_train.shape[2])
    elif architecture == "inception_lstm_calculus_fusion":
        model = InceptionLstmCalculusFusionClassifier(x_train.shape[2])
    else:
        raise ValueError(f"Unknown sequence architecture: {architecture}")
    if class_weighted:
        counts = np.bincount(y_train, minlength=3).astype(float)
        class_weights = counts.sum() / np.maximum(counts, 1.0)
        class_weights = class_weights / np.mean(class_weights)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    val_pred = _predict_torch_sequence_classifier(model, x_val)
    return model, val_pred


def _predict_torch_sequence_classifier(model: object, x: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 2048):
            xb = torch.tensor(x[start : start + 2048], dtype=torch.float32)
            preds.append(torch.argmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(preds)


def _tree_model_specs() -> list[tuple[str, object]]:
    specs: list[tuple[str, object]] = []
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        specs.extend(
            [
                (
                    "sklearn_hgbr_depth3",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        random_state=0,
                    ),
                ),
                (
                    "sklearn_hgbr_depth5",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.03,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=0,
                    ),
                ),
            ]
        )
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor

        specs.append(
            (
                "xgboost_depth2",
                XGBRegressor(
                    n_estimators=250,
                    max_depth=2,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=10.0,
                    reg_alpha=0.1,
                    objective="reg:squarederror",
                    random_state=0,
                    n_jobs=1,
                    verbosity=0,
                ),
            )
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        specs.append(
            (
                "lightgbm_depth3",
                LGBMRegressor(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=10.0,
                    reg_alpha=0.1,
                    random_state=0,
                    n_jobs=1,
                    verbose=-1,
                ),
            )
        )
    except Exception:
        pass
    return specs


def _model_result(
    *,
    feature_set: str,
    model_name: str,
    feature_count: int,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    details: dict[str, float],
) -> dict[str, object]:
    return {
        "feature_set": feature_set,
        "model_name": model_name,
        "feature_count": feature_count,
        "validation_rmse": validation_metrics["rmse"],
        "validation_qlike": validation_metrics["qlike"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "qlike": test_metrics["qlike"],
        "details": details,
    }


def _best_model_result(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("No model results available")
    return min(results, key=lambda row: float(row["validation_rmse"]))


def _ridge_coef(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-8)
    err = y_pred - y_true
    qlike = np.mean(np.log(y_pred * y_pred) + (y_true * y_true) / (y_pred * y_pred))
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "qlike": float(qlike),
    }


def _qlike_calibration_scale(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-8)
    scale = np.sqrt(np.mean((y_true / y_pred) ** 2))
    return float(np.clip(scale, 0.25, 4.0))


def _intraday_time_features(timestamps: np.ndarray) -> tuple[np.ndarray, list[str]]:
    minute_of_day = []
    day_of_week = []
    for timestamp in timestamps:
        dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        minute_of_day.append(dt.hour * 60 + dt.minute)
        day_of_week.append(dt.weekday())
    minute = np.asarray(minute_of_day, dtype=float)
    dow = np.asarray(day_of_week, dtype=float)
    minute_angle = 2.0 * np.pi * minute / (24.0 * 60.0)
    dow_angle = 2.0 * np.pi * dow / 5.0
    features = np.stack(
        [
            np.sin(minute_angle),
            np.cos(minute_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
        ],
        axis=1,
    )
    return features, ["minute_sin", "minute_cos", "dow_sin", "dow_cos"]


def _parse_book_levels(value: object) -> list[dict[str, float]]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, list):
        parsed = value
    else:
        parsed = ast.literal_eval(str(value))
    return [{"price": float(level["price"]), "size": float(level["size"])} for level in parsed]


def _depth(levels: list[dict[str, float]], n: int) -> float:
    return float(sum(level["size"] for level in levels[:n]))


def _weighted_price(levels: list[dict[str, float]], n: int) -> float:
    size = _depth(levels, n)
    if size <= 1e-12:
        return 0.0
    return float(sum(level["price"] * level["size"] for level in levels[:n]) / size)


def _polymarket_standard_feature_names() -> list[str]:
    return [
        "mid_price",
        "spread",
        "bid_depth_1",
        "ask_depth_1",
        "bid_depth_3",
        "ask_depth_3",
        "bid_depth_10",
        "ask_depth_10",
        "imbalance_1",
        "imbalance_10",
        "microprice_1",
        "weighted_bid_10",
        "weighted_ask_10",
    ]


def _polymarket_standard_row(
    *,
    mid: float,
    spread: float,
    bid_levels: list[dict[str, float]],
    ask_levels: list[dict[str, float]],
) -> list[float]:
    bid_1 = _depth(bid_levels, 1)
    ask_1 = _depth(ask_levels, 1)
    bid_3 = _depth(bid_levels, 3)
    ask_3 = _depth(ask_levels, 3)
    bid_10 = _depth(bid_levels, 10)
    ask_10 = _depth(ask_levels, 10)
    best_bid = bid_levels[0]["price"]
    best_ask = ask_levels[0]["price"]
    imbalance_1 = (bid_1 - ask_1) / (bid_1 + ask_1 + 1e-12)
    imbalance_10 = (bid_10 - ask_10) / (bid_10 + ask_10 + 1e-12)
    microprice_1 = (best_ask * bid_1 + best_bid * ask_1) / (bid_1 + ask_1 + 1e-12)
    return [
        float(mid),
        float(spread),
        bid_1,
        ask_1,
        bid_3,
        ask_3,
        bid_10,
        ask_10,
        float(imbalance_1),
        float(imbalance_10),
        float(microprice_1),
        _weighted_price(bid_levels, 10),
        _weighted_price(ask_levels, 10),
    ]


def _polymarket_calculus_feature_names() -> list[str]:
    return [
        f"{series}_jet_{lookback}_{name}"
        for series in ("mid", "spread", "imbalance")
        for lookback in (6, 18)
        for name in ("slope", "curvature", "roughness")
    ]


def _polymarket_calculus_row(
    mids: np.ndarray,
    spreads: np.ndarray,
    imbalance: np.ndarray,
    idx: int,
) -> list[float]:
    row = []
    for series in (mids, spreads, imbalance):
        for lookback in (6, 18):
            jets = _local_taylor_jets(series[idx - lookback + 1 : idx + 1])
            row.extend([float(jets[1]), float(jets[2]), float(jets[4])])
    return row


def _yahoo_chart_url(*, symbol: str, start: str, end: str) -> str:
    period1 = _date_to_epoch(start)
    period2 = _date_to_epoch(end) + 24 * 60 * 60
    encoded_symbol = quote(symbol.upper(), safe="")
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )


def _yahoo_intraday_url(*, symbol: str, intraday_range: str, interval: str) -> str:
    encoded_symbol = quote(symbol.upper(), safe="")
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?range={intraday_range}&interval={interval}&includePrePost=false&events=history"
    )


def _date_to_epoch(date_text: str) -> int:
    return int(datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _parse_yahoo_chart(*, symbol: str, payload: dict[str, object]) -> MarketOHLCV:
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("Yahoo payload missing chart object")
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo chart error: {error}")
    result = chart.get("result")
    if not isinstance(result, list) or not result:
        raise ValueError("Yahoo payload has no chart result")
    data = result[0]
    if not isinstance(data, dict):
        raise ValueError("Yahoo chart result has invalid shape")
    timestamps = np.asarray(data.get("timestamp", []), dtype=float)
    indicators = data.get("indicators")
    if not isinstance(indicators, dict):
        raise ValueError("Yahoo payload missing indicators")
    quote_list = indicators.get("quote")
    adj_list = indicators.get("adjclose")
    if not isinstance(quote_list, list) or not quote_list:
        raise ValueError("Yahoo payload missing quote data")
    quote = quote_list[0]
    adj = adj_list[0] if isinstance(adj_list, list) and adj_list else None
    if not isinstance(quote, dict):
        raise ValueError("Yahoo quote data has invalid shape")

    open_ = _json_float_array(quote.get("open"))
    high = _json_float_array(quote.get("high"))
    low = _json_float_array(quote.get("low"))
    close = _json_float_array(quote.get("close"))
    volume = _json_float_array(quote.get("volume"))
    adj_close = _json_float_array(adj.get("adjclose")) if isinstance(adj, dict) else close
    mask = (
        np.isfinite(timestamps)
        & np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(adj_close)
        & np.isfinite(volume)
        & (close > 0.0)
        & (adj_close > 0.0)
    )
    if int(np.sum(mask)) < 300:
        raise ValueError("Yahoo payload has too few valid observations")
    return MarketOHLCV(
        symbol=symbol.upper(),
        timestamp=timestamps[mask],
        open=open_[mask],
        high=high[mask],
        low=low[mask],
        close=close[mask],
        adj_close=adj_close[mask],
        volume=volume[mask],
    )


def _json_float_array(values: object) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError("Expected JSON list")
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def _timestamp_dates(timestamps: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date().toordinal()
            for timestamp in timestamps
        ],
        dtype=int,
    )


def _gate_analysis(
    *,
    baseline_name: str,
    candidate_name: str,
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    passed: bool,
    improvement: float,
) -> list[str]:
    if passed:
        return [
            f"{candidate_name} beat {baseline_name} by {improvement:.2%} RMSE.",
            "QLIKE was not worse, so the gain is not just an RMSE artifact.",
            "Proceed to the next milestone only after confirming the data cache and split are leak-free.",
        ]
    reasons = []
    if improvement < 0.0:
        reasons.append(f"{candidate_name} underperformed {baseline_name} on RMSE.")
    elif improvement < 0.01:
        reasons.append(f"{candidate_name} improved RMSE by only {improvement:.2%}, below the gate.")
    if candidate_metrics["qlike"] > baseline_metrics["qlike"]:
        reasons.append("QLIKE worsened, which means variance calibration degraded.")
    reasons.append(
        "Root-cause candidates: jets may be too noisy on daily data, HAR may already absorb the useful scales, "
        "or the target horizon may need intraday realized volatility rather than close-to-close proxy volatility."
    )
    return reasons


def _robustness_analysis(*, target_results: list[dict[str, object]], passed: bool) -> list[str]:
    passed_targets = [str(row["target_symbol"]) for row in target_results if row["passed"]]
    failed_targets = [str(row["target_symbol"]) for row in target_results if not row["passed"]]
    if passed:
        return [
            f"Robustness gate passed on {len(passed_targets)}/{len(target_results)} targets.",
            f"Passed targets: {', '.join(passed_targets)}.",
            "The daily OHLCV track is strong enough to justify a separate intraday/LOB milestone.",
        ]
    return [
        f"Robustness gate failed on targets: {', '.join(failed_targets)}.",
        "Root-cause candidates: cross-asset information may be target-specific, the asset universe is too equity-heavy, "
        "or daily close-to-close volatility is too coarse for universal transfer.",
    ]


def _sota_candidate_analysis(*, target_results: list[dict[str, object]], passed: bool) -> list[str]:
    passed_targets = [str(row["target_symbol"]) for row in target_results if row["passed"]]
    failed_targets = [str(row["target_symbol"]) for row in target_results if not row["passed"]]
    best_pairs = [
        f"{row['target_symbol']}: {row['best_candidate']['model_name']} over {row['best_baseline']['model_name']}"
        for row in target_results
    ]
    if passed:
        return [
            f"SOTA-candidate gate passed on {len(passed_targets)}/{len(target_results)} targets.",
            f"Passed targets: {', '.join(passed_targets)}.",
            "Best candidate/baseline pairs: " + "; ".join(best_pairs) + ".",
            "This supports the daily-data calculus-feature claim, but not a universal finance SOTA claim.",
        ]
    return [
        f"SOTA-candidate gate failed on targets: {', '.join(failed_targets)}.",
        "Best candidate/baseline pairs: " + "; ".join(best_pairs) + ".",
        "Root-cause candidates: boosted non-jet features may already capture the daily signal, "
        "validation-selected candidate models may be over-regularized, or daily close-to-close data may be near its ceiling.",
    ]


def _intraday_analysis(*, target_results: list[dict[str, object]], passed: bool) -> list[str]:
    passed_targets = [str(row["target_symbol"]) for row in target_results if row["passed"]]
    failed_targets = [str(row["target_symbol"]) for row in target_results if not row["passed"]]
    best_pairs = [
        f"{row['target_symbol']}: {row['best_candidate']['model_name']} over {row['best_baseline']['model_name']}"
        for row in target_results
    ]
    if passed:
        return [
            f"Intraday gate passed on {len(passed_targets)}/{len(target_results)} targets.",
            f"Passed targets: {', '.join(passed_targets)}.",
            "Best candidate/baseline pairs: " + "; ".join(best_pairs) + ".",
        ]
    return [
        f"Intraday gate failed on targets: {', '.join(failed_targets)}.",
        "Best candidate/baseline pairs: " + "; ".join(best_pairs) + ".",
        "Root-cause candidates: 5-minute Yahoo data is short and noisy, time-of-day controls may dominate, "
        "or true short-horizon edge may require order-book imbalance rather than OHLCV bars.",
    ]


def _orderbook_analysis(*, target_results: list[dict[str, object]], passed: bool) -> list[str]:
    passed_assets = [str(row["asset"]) for row in target_results if row["passed"]]
    failed_assets = [str(row["asset"]) for row in target_results if not row["passed"]]
    best_pairs = [
        f"{row['asset']}: {row['best_candidate']['model_name']} over {row['best_baseline']['model_name']}"
        for row in target_results
    ]
    if passed:
        return [
            f"Order-book gate passed on {len(passed_assets)}/{len(target_results)} assets.",
            f"Passed assets: {', '.join(passed_assets)}.",
            "Best candidate/baseline pairs: " + "; ".join(best_pairs) + ".",
            "This is a real order-book signal, but it is Polymarket-specific and not yet FI-2010/DeepLOB SOTA.",
        ]
    return [
        f"Order-book gate failed on assets: {', '.join(failed_assets)}.",
        "Best candidate/baseline pairs: " + "; ".join(best_pairs) + ".",
        "Root-cause candidates: binary-market order books have special boundary dynamics, "
        "the horizon may be wrong, or sequence models may be needed to beat boosted depth features.",
    ]


def _fi2010_analysis(
    *,
    baseline: dict[str, object],
    candidate: dict[str, object],
    passed: bool,
) -> list[str]:
    if passed:
        return [
            f"Candidate {candidate['model_name']} beat baseline {baseline['model_name']} on FI-2010 subset accuracy.",
            "Macro-F1 improved or stayed flat, so the gain is not only from majority-class accuracy.",
            "This is closer to a canonical LOB benchmark, but still not a full DeepLOB SOTA reproduction.",
        ]
    return [
        f"Candidate {candidate['model_name']} did not beat baseline {baseline['model_name']} under the strict gate.",
        "Root-cause candidates: handcrafted jets may need sequence models, the label horizon may differ, "
        "or the subset size may be too small for stable FI-2010 comparison.",
    ]


def _fi2010_sequence_analysis(
    *,
    baseline: dict[str, object],
    candidate: dict[str, object],
    passed: bool,
) -> list[str]:
    if passed:
        return [
            "The same CNN-GRU sequence architecture improved when calculus channels were added.",
            f"Baseline macro-F1 `{baseline['macro_f1']:.4f}` vs candidate `{candidate['macro_f1']:.4f}`.",
            "This is the strongest evidence so far that the calculus features can complement neural LOB models.",
        ]
    return [
        "The CNN-GRU sequence gate did not pass.",
        f"Baseline macro-F1 `{baseline['macro_f1']:.4f}` vs candidate `{candidate['macro_f1']:.4f}`.",
        "Root-cause candidates: the model may be undertrained on CPU, sequence length may be wrong, "
        "or the calculus channels need to be learned inside the network rather than precomputed.",
    ]


def _fi2010_deeplob_analysis(
    *,
    baseline: dict[str, object],
    candidate: dict[str, object],
    passed: bool,
) -> list[str]:
    if passed:
        return [
            "The Inception-LSTM sequence architecture improved when calculus channels were added.",
            f"Baseline macro-F1 `{baseline['macro_f1']:.4f}` vs candidate `{candidate['macro_f1']:.4f}`.",
            "This is the closest experiment so far to a DeepLOB-style calculus-channel ablation.",
        ]
    return [
        "The Inception-LSTM calculus-channel gate did not pass.",
        f"Baseline macro-F1 `{baseline['macro_f1']:.4f}` vs candidate `{candidate['macro_f1']:.4f}`.",
        "Root-cause candidates: the compact CPU architecture is still not full DeepLOB, "
        "training may be under-budgeted, or precomputed calculus channels need architecture-specific normalization.",
    ]
