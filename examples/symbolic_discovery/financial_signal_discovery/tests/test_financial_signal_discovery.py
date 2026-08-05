# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
from __future__ import annotations

import json
import zipfile

import numpy as np

from examples.symbolic_discovery.financial_signal_discovery import benchmark as fsd_benchmark
from examples.symbolic_discovery.financial_signal_discovery.benchmark import (
    _chronological_split,
    _fit_tuned_linear,
    _parse_yahoo_chart,
    align_market_adj_close,
    align_market_returns,
    build_cross_asset_compact_calculus_features,
    build_cross_asset_features,
    build_cross_asset_sota_candidate_features,
    build_cross_asset_standard_features,
    build_fi2010_calculus_features,
    build_fi2010_compact_calculus_features,
    build_har_features,
    build_intraday_compact_calculus_features,
    build_intraday_standard_features,
    build_jet_features,
    build_multiscale_jet_features,
    build_polymarket_orderbook_features,
    evaluate_milestone1,
    evaluate_milestone2,
    evaluate_milestone3,
    evaluate_milestone4,
    evaluate_milestone5,
    evaluate_milestone6,
    evaluate_milestone7,
    evaluate_milestone8,
    evaluate_milestone9,
    evaluate_milestone10,
    fetch_yahoo_intraday_ohlcv,
    fetch_yahoo_ohlcv,
    intraday_volatility_dataset_from_ohlcv,
    load_fi2010_matrix,
    make_sequence_windows,
    make_synthetic_volatility_dataset,
    market_volatility_dataset_from_ohlcv,
)


def test_synthetic_dataset_has_positive_volatility_targets() -> None:
    dataset = make_synthetic_volatility_dataset(n_days=600, seed=7)

    assert dataset.returns.ndim == 1
    assert np.all(dataset.latent_vol > 0.0)
    assert np.all(dataset.observed_vol >= 0.0)
    assert np.all(dataset.target_vol[:-dataset.horizon] > 0.0)


def test_jet_features_extend_har_features_without_target_mismatch() -> None:
    dataset = make_synthetic_volatility_dataset(n_days=700, seed=3)
    har_x, har_y, har_names = build_har_features(dataset)
    jet_x, jet_y, jet_names = build_jet_features(dataset)

    assert har_names == ["rv_1", "rv_5", "rv_22"]
    assert jet_names[: len(har_names)] == har_names
    assert jet_x.shape[0] == har_x.shape[0]
    assert jet_x.shape[1] > har_x.shape[1]
    assert np.allclose(jet_y, har_y)
    assert np.isfinite(jet_x).all()


def test_chronological_split_keeps_time_order() -> None:
    x = np.arange(100, dtype=float).reshape(50, 2)
    y = np.arange(50, dtype=float)
    split = _chronological_split(x, y, ["a", "b"])

    assert split.y_train[-1] < split.y_val[0]
    assert split.y_val[-1] < split.y_test[0]
    assert split.x_train.shape[0] == 30
    assert split.x_val.shape[0] == 10
    assert split.x_test.shape[0] == 10


def test_log_volatility_model_predicts_positive_values() -> None:
    dataset = make_synthetic_volatility_dataset(n_days=700, seed=5)
    split = _chronological_split(*build_jet_features(dataset))
    model = _fit_tuned_linear(split)
    pred = model.predict(split.x_test)

    assert model.target_transform == "log"
    assert np.all(pred > 0.0)
    assert np.isfinite(pred).all()


def test_milestone1_success_gate_passes() -> None:
    results = evaluate_milestone1(n_days=1200, seed=0)
    gate = results["success_gate"]
    models = results["models"]

    assert gate["passed"] is True
    assert gate["rmse_improvement"] >= gate["rmse_improvement_required"]
    assert gate["qlike_not_worse"] is True
    assert models["jet_augmented"]["rmse"] < models["har_baseline"]["rmse"]


def test_yahoo_chart_parser_filters_invalid_rows() -> None:
    payload = _fake_yahoo_payload(n=360)
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][10] = None
    parsed = _parse_yahoo_chart(symbol="spy", payload=payload)

    assert parsed.symbol == "SPY"
    assert parsed.close.size == 359
    assert np.all(parsed.adj_close > 0.0)


def test_yahoo_chart_parser_uses_close_when_intraday_has_no_adjclose() -> None:
    payload = _fake_intraday_yahoo_payload(n=500)
    parsed = _parse_yahoo_chart(symbol="SPY", payload=payload)

    assert parsed.symbol == "SPY"
    assert parsed.close.size == 500
    assert np.allclose(parsed.adj_close, parsed.close)


def test_market_volatility_dataset_is_leakage_aligned() -> None:
    ohlcv = _parse_yahoo_chart(symbol="SPY", payload=_fake_yahoo_payload(n=420))
    dataset = market_volatility_dataset_from_ohlcv(ohlcv, horizon=5)
    t = dataset.feature_start

    expected_target = np.sqrt(np.mean(dataset.returns[t + 1 : t + 6] ** 2)) * np.sqrt(252.0)
    assert np.isclose(dataset.target_vol[t], expected_target)
    assert dataset.observed_vol[t] != dataset.target_vol[t]


def test_multiscale_jet_features_extend_har_for_m2() -> None:
    dataset = make_synthetic_volatility_dataset(n_days=700, seed=11)
    x, y, names = build_multiscale_jet_features(dataset, lookbacks=(11, 22))

    assert names[:3] == ["rv_1", "rv_5", "rv_22"]
    assert "log_rv_11_slope" in names
    assert "log_rv_22_curvature" in names
    assert x.shape[1] == len(names)
    assert x.shape[0] == y.shape[0]


def test_fetch_yahoo_ohlcv_uses_cache_without_network(tmp_path) -> None:
    cache_path = tmp_path / "SPY_2005-01-01_2025-12-31_yahoo_chart.json"
    cache_path.write_text(__import__("json").dumps(_fake_yahoo_payload(n=360)))

    parsed = fetch_yahoo_ohlcv(cache_dir=tmp_path)

    assert parsed.symbol == "SPY"
    assert parsed.close.size == 360


def test_milestone2_can_run_from_cached_market_data(tmp_path) -> None:
    cache_path = tmp_path / "SPY_2005-01-01_2025-12-31_yahoo_chart.json"
    cache_path.write_text(__import__("json").dumps(_fake_yahoo_payload(n=900)))

    results = evaluate_milestone2(cache_dir=tmp_path)
    gate = results["success_gate"]

    assert results["milestone"] == "M2 real SPY volatility jets"
    assert "passed" in gate
    assert np.isfinite(gate["rmse_improvement"])


def test_cross_asset_alignment_and_features_are_leakage_safe() -> None:
    markets = [
        _parse_yahoo_chart(symbol="SPY", payload=_fake_yahoo_payload(n=500, seed=1)),
        _parse_yahoo_chart(symbol="QQQ", payload=_fake_yahoo_payload(n=500, seed=2)),
        _parse_yahoo_chart(symbol="TLT", payload=_fake_yahoo_payload(n=500, seed=3)),
    ]
    aligned = align_market_returns(markets)
    levels = align_market_adj_close(markets)
    x, y, names = build_cross_asset_features(
        returns=aligned,
        levels=levels,
        symbols=("SPY", "QQQ", "TLT"),
        lookbacks=(11, 22),
    )

    assert aligned.shape == (499, 3)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == len(names)
    assert "QQQ_log_rv_11_slope" in names
    assert "QQQ_log_level_11_slope" in names
    assert "TLT_rv_22" in names


def test_cross_asset_standard_features_exclude_jet_names() -> None:
    markets = [
        _parse_yahoo_chart(symbol="SPY", payload=_fake_yahoo_payload(n=500, seed=1)),
        _parse_yahoo_chart(symbol="QQQ", payload=_fake_yahoo_payload(n=500, seed=2)),
    ]
    aligned = align_market_returns(markets)
    levels = align_market_adj_close(markets)
    x, y, names = build_cross_asset_standard_features(
        returns=aligned,
        levels=levels,
        symbols=("SPY", "QQQ"),
    )

    assert x.shape[0] == y.shape[0]
    assert "QQQ_rv_22" in names
    assert "QQQ_log_level_22" in names
    assert not any("roughness" in name or "curvature" in name for name in names)


def test_sota_candidate_features_are_standard_plus_jets() -> None:
    markets = [
        _parse_yahoo_chart(symbol="SPY", payload=_fake_yahoo_payload(n=500, seed=1)),
        _parse_yahoo_chart(symbol="QQQ", payload=_fake_yahoo_payload(n=500, seed=2)),
    ]
    aligned = align_market_returns(markets)
    levels = align_market_adj_close(markets)
    standard_x, standard_y, standard_names = build_cross_asset_standard_features(
        returns=aligned,
        levels=levels,
        symbols=("SPY", "QQQ"),
    )
    candidate_x, candidate_y, candidate_names = build_cross_asset_sota_candidate_features(
        returns=aligned,
        levels=levels,
        symbols=("SPY", "QQQ"),
    )

    assert candidate_x.shape[0] == standard_x.shape[0]
    assert candidate_x.shape[1] > standard_x.shape[1]
    assert np.allclose(candidate_y, standard_y)
    assert candidate_names[: len(standard_names)] == standard_names
    assert any(name.startswith("jet__") for name in candidate_names)


def test_compact_calculus_features_keep_only_roughness_jets() -> None:
    markets = [
        _parse_yahoo_chart(symbol="SPY", payload=_fake_yahoo_payload(n=500, seed=1)),
        _parse_yahoo_chart(symbol="QQQ", payload=_fake_yahoo_payload(n=500, seed=2)),
    ]
    aligned = align_market_returns(markets)
    levels = align_market_adj_close(markets)
    _, _, names = build_cross_asset_compact_calculus_features(
        returns=aligned,
        levels=levels,
        symbols=("SPY", "QQQ"),
    )
    jet_names = [name for name in names if name.startswith("jet__")]

    assert jet_names
    assert all("roughness" in name for name in jet_names)


def test_milestone3_can_run_from_cached_market_data(tmp_path) -> None:
    for idx, symbol in enumerate(("SPY", "^VIX", "QQQ", "IWM", "TLT", "GLD")):
        cache_path = tmp_path / f"{symbol}_2005-01-01_2025-12-31_yahoo_chart.json"
        cache_path.write_text(__import__("json").dumps(_fake_yahoo_payload(n=900, seed=idx + 1)))

    results = evaluate_milestone3(cache_dir=tmp_path)
    gate = results["success_gate"]

    assert results["milestone"] == "M3 cross-asset volatility context"
    assert "passed" in gate
    assert np.isfinite(gate["rmse_improvement"])


def test_milestone4_aggregates_multiple_targets_from_cache(tmp_path) -> None:
    for idx, symbol in enumerate(("SPY", "^VIX", "QQQ", "IWM", "TLT", "GLD")):
        cache_path = tmp_path / f"{symbol}_2005-01-01_2025-12-31_yahoo_chart.json"
        cache_path.write_text(__import__("json").dumps(_fake_yahoo_payload(n=900, seed=idx + 10)))

    results = evaluate_milestone4(
        target_symbols=("SPY", "QQQ"),
        context_universe=("^VIX", "SPY", "QQQ", "IWM", "TLT", "GLD"),
        cache_dir=tmp_path,
    )
    gate = results["success_gate"]

    assert results["milestone"] == "M4 multi-target robustness"
    assert len(results["targets"]) == 2
    assert "passed" in gate
    assert np.isfinite(gate["median_rmse_improvement"])


def test_milestone5_model_zoo_runs_from_cache_with_optional_boosters_disabled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(fsd_benchmark, "_tree_model_specs", lambda: [])
    for idx, symbol in enumerate(("SPY", "^VIX", "QQQ", "IWM", "TLT", "GLD")):
        cache_path = tmp_path / f"{symbol}_2005-01-01_2025-12-31_yahoo_chart.json"
        cache_path.write_text(__import__("json").dumps(_fake_yahoo_payload(n=900, seed=idx + 20)))

    results = evaluate_milestone5(
        target_symbols=("SPY",),
        context_universe=("^VIX", "SPY", "QQQ", "IWM", "TLT", "GLD"),
        cache_dir=tmp_path,
    )
    target = results["targets"][0]

    assert results["milestone"] == "M5 daily SOTA-candidate model zoo"
    assert target["best_baseline"]["model_name"] == "ridge_logvol"
    assert str(target["best_candidate"]["feature_set"]).startswith("cross_asset")
    assert np.isfinite(target["rmse_improvement"])


def test_intraday_features_are_leakage_aligned() -> None:
    market = _parse_yahoo_chart(symbol="SPY", payload=_fake_intraday_yahoo_payload(n=900))
    dataset = intraday_volatility_dataset_from_ohlcv(market, horizon=6)
    timestamps = market.timestamp[1:]
    x_standard, y_standard, standard_names = build_intraday_standard_features(
        dataset=dataset,
        timestamps=timestamps,
    )
    x_candidate, y_candidate, candidate_names = build_intraday_compact_calculus_features(
        dataset=dataset,
        timestamps=timestamps,
    )
    t = dataset.feature_start
    expected_target = np.sqrt(np.mean(dataset.returns[t + 1 : t + 7] ** 2)) * np.sqrt(252.0 * 78.0)

    assert np.isclose(dataset.target_vol[t], expected_target)
    assert x_candidate.shape[0] == x_standard.shape[0]
    assert x_candidate.shape[1] > x_standard.shape[1]
    assert np.allclose(y_candidate, y_standard)
    assert "minute_sin" in standard_names
    assert "jet_log_rv_78_roughness" in candidate_names


def test_fetch_yahoo_intraday_ohlcv_uses_cache_without_network(tmp_path) -> None:
    cache_path = tmp_path / "SPY_60d_5m_yahoo_intraday.json"
    cache_path.write_text(__import__("json").dumps(_fake_intraday_yahoo_payload(n=500)))

    parsed = fetch_yahoo_intraday_ohlcv(cache_dir=tmp_path)

    assert parsed.symbol == "SPY"
    assert parsed.close.size == 500


def test_milestone6_runs_from_cached_intraday_data_with_optional_boosters_disabled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(fsd_benchmark, "_tree_model_specs", lambda: [])
    for idx, symbol in enumerate(("SPY", "QQQ")):
        cache_path = tmp_path / f"{symbol}_60d_5m_yahoo_intraday.json"
        cache_path.write_text(__import__("json").dumps(_fake_intraday_yahoo_payload(n=900, seed=idx + 30)))

    results = evaluate_milestone6(target_symbols=("SPY", "QQQ"), cache_dir=tmp_path)

    assert results["milestone"] == "M6 intraday short-horizon volatility"
    assert len(results["targets"]) == 2
    assert np.isfinite(results["success_gate"]["median_rmse_improvement"])


def test_polymarket_orderbook_features_add_calculus_jets() -> None:
    orderbook = _fake_polymarket_orderbook(n_groups=16, n_per_group=70)
    x_standard, y_standard, standard_names = build_polymarket_orderbook_features(
        orderbook=orderbook,
        asset="BTC",
        horizon=6,
        include_calculus=False,
    )
    x_calculus, y_calculus, calculus_names = build_polymarket_orderbook_features(
        orderbook=orderbook,
        asset="BTC",
        horizon=6,
        include_calculus=True,
    )

    assert x_calculus.shape[0] == x_standard.shape[0]
    assert x_calculus.shape[1] > x_standard.shape[1]
    assert np.allclose(y_calculus, y_standard)
    assert calculus_names[: len(standard_names)] == standard_names
    assert "mid_jet_18_roughness" in calculus_names


def test_milestone7_runs_from_fake_orderbook_with_optional_boosters_disabled(monkeypatch) -> None:
    monkeypatch.setattr(fsd_benchmark, "_tree_model_specs", lambda: [])
    monkeypatch.setattr(
        fsd_benchmark,
        "fetch_polymarket_orderbook",
        lambda **_: _fake_polymarket_orderbook(n_groups=20, n_per_group=80),
    )

    results = evaluate_milestone7(assets=("BTC", "ETH"), horizon=6, max_samples_per_asset=2_000)

    assert results["milestone"] == "M7 real order-book microstructure"
    assert len(results["targets"]) == 2
    assert np.isfinite(results["success_gate"]["median_rmse_improvement"])


def test_fi2010_loader_and_calculus_features(tmp_path) -> None:
    zip_path = tmp_path / "fi2010.zip"
    matrix = _fake_fi2010_matrix(n=200)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("train.txt", "\n".join(" ".join(map(str, row)) for row in matrix))

    x, y = load_fi2010_matrix(zip_path=zip_path, member="train.txt", max_samples=150)
    x_calc = build_fi2010_calculus_features(x)
    x_compact = build_fi2010_compact_calculus_features(x)

    assert x.shape == (150, 40)
    assert y.shape == (150,)
    assert set(np.unique(y)).issubset({0, 1, 2})
    assert x_calc.shape[0] == x.shape[0]
    assert x_calc.shape[1] == 49
    assert x_compact.shape == (150, 43)


def test_milestone8_runs_from_fake_fi2010_zip(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "fi2010.zip"
    train = _fake_fi2010_matrix(n=260)
    test = _fake_fi2010_matrix(n=180)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Train_Dst_NoAuction_DecPre_CF_7.txt", "\n".join(" ".join(map(str, row)) for row in train))
        archive.writestr("Test_Dst_NoAuction_DecPre_CF_7.txt", "\n".join(" ".join(map(str, row)) for row in test))
    monkeypatch.setattr(fsd_benchmark, "fetch_deeplob_fi2010_zip", lambda **_: zip_path)

    results = evaluate_milestone8(max_train_samples=240, max_test_samples=160)

    assert results["milestone"] == "M8 FI-2010 DeepLOB-style classification"
    assert "passed" in results["success_gate"]
    assert np.isfinite(results["success_gate"]["accuracy_improvement"])


def test_fi2010_sequence_windows_are_chronological() -> None:
    x = np.arange(60, dtype=float).reshape(20, 3)
    y = np.arange(20, dtype=int) % 3
    windows, labels = make_sequence_windows(x, y, sequence_window=5)

    assert windows.shape == (16, 5, 3)
    assert np.allclose(windows[0], x[:5])
    assert np.allclose(windows[-1], x[-5:])
    assert np.array_equal(labels, y[4:])


def test_milestone9_sequence_model_runs_from_fake_fi2010_zip(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "fi2010.zip"
    train = _fake_fi2010_matrix(n=180)
    test = _fake_fi2010_matrix(n=120)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Train_Dst_NoAuction_DecPre_CF_7.txt", "\n".join(" ".join(map(str, row)) for row in train))
        archive.writestr("Test_Dst_NoAuction_DecPre_CF_7.txt", "\n".join(" ".join(map(str, row)) for row in test))
    monkeypatch.setattr(fsd_benchmark, "fetch_deeplob_fi2010_zip", lambda **_: zip_path)

    results = evaluate_milestone9(
        max_train_samples=160,
        max_test_samples=100,
        sequence_window=10,
        epochs=1,
    )

    assert results["milestone"] == "M9 FI-2010 neural sequence calculus channels"
    assert "passed" in results["success_gate"]
    assert np.isfinite(results["success_gate"]["accuracy_improvement"])


def test_milestone10_inception_lstm_runs_from_fake_fi2010_zip(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "fi2010.zip"
    train = _fake_fi2010_matrix(n=180)
    test = _fake_fi2010_matrix(n=120)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Train_Dst_NoAuction_DecPre_CF_7.txt", "\n".join(" ".join(map(str, row)) for row in train))
        archive.writestr("Test_Dst_NoAuction_DecPre_CF_7.txt", "\n".join(" ".join(map(str, row)) for row in test))
    monkeypatch.setattr(fsd_benchmark, "fetch_deeplob_fi2010_zip", lambda **_: zip_path)

    results = evaluate_milestone10(
        max_train_samples=160,
        max_test_samples=100,
        sequence_window=10,
        epochs=1,
    )

    assert results["milestone"] == "M10 FI-2010 DeepLOB Inception-LSTM calculus channels"
    assert results["models"]["best_baseline"]["architecture"] == "inception_lstm"
    assert "passed" in results["success_gate"]
    assert np.isfinite(results["success_gate"]["accuracy_improvement"])


def _fake_yahoo_payload(*, n: int, seed: int = 123) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    returns = 0.0002 + 0.01 * rng.normal(size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * 1.01
    low = close * 0.99
    open_ = close / (1.0 + 0.001 * rng.normal(size=n))
    volume = np.full(n, 10_000_000.0)
    return {
        "chart": {
            "result": [
                {
                    "timestamp": list(range(1_100_000_000, 1_100_000_000 + n * 86_400, 86_400)),
                    "indicators": {
                        "quote": [
                            {
                                "open": open_.tolist(),
                                "high": high.tolist(),
                                "low": low.tolist(),
                                "close": close.tolist(),
                                "volume": volume.tolist(),
                            }
                        ],
                        "adjclose": [{"adjclose": close.tolist()}],
                    },
                }
            ],
            "error": None,
        }
    }


def _fake_intraday_yahoo_payload(*, n: int, seed: int = 321) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    returns = 0.00001 + 0.001 * rng.normal(size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * 1.0005
    low = close * 0.9995
    open_ = close / (1.0 + 0.0001 * rng.normal(size=n))
    volume = np.full(n, 100_000.0)
    return {
        "chart": {
            "result": [
                {
                    "timestamp": list(range(1_700_000_000, 1_700_000_000 + n * 300, 300)),
                    "indicators": {
                        "quote": [
                            {
                                "open": open_.tolist(),
                                "high": high.tolist(),
                                "low": low.tolist(),
                                "close": close.tolist(),
                                "volume": volume.tolist(),
                            }
                        ],
                    },
                }
            ],
            "error": None,
        }
    }


def _fake_polymarket_orderbook(*, n_groups: int, n_per_group: int):
    import pandas as pd

    rows = []
    assets = ("BTC", "ETH")
    for group_idx in range(n_groups):
        asset = assets[group_idx % len(assets)]
        token_id = f"{asset}_{group_idx}"
        market_id = f"{asset}_market_{group_idx}"
        phase = 0.1 * group_idx
        for idx in range(n_per_group):
            mid = 0.5 + 0.08 * np.sin(idx / 8.0 + phase)
            spread = 0.02 + 0.005 * np.cos(idx / 11.0)
            best_bid = max(0.01, mid - spread / 2.0)
            best_ask = min(0.99, mid + spread / 2.0)
            bid_levels = [
                {"price": max(0.01, best_bid - level * 0.01), "size": 1000.0 + 10.0 * idx + level}
                for level in range(10)
            ]
            ask_levels = [
                {"price": min(0.99, best_ask + level * 0.01), "size": 900.0 + 8.0 * idx + level}
                for level in range(10)
            ]
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-03-06") + pd.Timedelta(seconds=10 * idx),
                    "asset": asset,
                    "market_id": market_id,
                    "condition_id": market_id,
                    "token_id": token_id,
                    "question": "fake",
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "mid_price": mid,
                    "bid_levels": json.dumps(bid_levels),
                    "ask_levels": json.dumps(ask_levels),
                }
            )
    return pd.DataFrame(rows)


def _fake_fi2010_matrix(*, n: int) -> np.ndarray:
    rng = np.random.default_rng(456)
    matrix = rng.normal(0.0, 0.01, size=(149, n))
    base = 0.25 + np.cumsum(rng.normal(0.0, 0.0005, size=n))
    for level in range(10):
        ask_price = base + 0.0001 * (level + 1)
        bid_price = base - 0.0001 * (level + 1)
        ask_size = 0.001 + 0.0001 * rng.random(n)
        bid_size = 0.001 + 0.0001 * rng.random(n)
        offset = 4 * level
        matrix[offset] = ask_price
        matrix[offset + 1] = ask_size
        matrix[offset + 2] = bid_price
        matrix[offset + 3] = bid_size
    labels = np.ones(n)
    labels[np.r_[False, np.diff(base) > 0.0001]] = 3
    labels[np.r_[False, np.diff(base) < -0.0001]] = 1
    labels[labels == 1] = 2
    labels[base > np.quantile(base, 0.66)] = 3
    labels[base < np.quantile(base, 0.33)] = 1
    for idx in range(5):
        matrix[144 + idx] = labels
    return matrix


def _fake_intraday_yahoo_payload(*, n: int, seed: int = 321) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    returns = 0.00001 + 0.001 * rng.normal(size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + 0.0005)
    low = close * (1.0 - 0.0005)
    open_ = close / (1.0 + 0.0001 * rng.normal(size=n))
    volume = np.full(n, 100_000.0)
    return {
        "chart": {
            "result": [
                {
                    "timestamp": list(range(1_700_000_000, 1_700_000_000 + n * 300, 300)),
                    "indicators": {
                        "quote": [
                            {
                                "open": open_.tolist(),
                                "high": high.tolist(),
                                "low": low.tolist(),
                                "close": close.tolist(),
                                "volume": volume.tolist(),
                            }
                        ],
                    },
                }
            ],
            "error": None,
        }
    }
