# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run financial signal discovery Milestone 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .benchmark import (
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
        write_artifacts,
    )
except ImportError:  # pragma: no cover
    from benchmark import (
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
        write_artifacts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/financial_signal_discovery"))
    parser.add_argument(
        "--milestone",
        choices=("m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"),
        default="m1",
    )
    parser.add_argument("--n-days", type=int, default=2600)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--context-symbols", default="^VIX,QQQ,IWM,TLT,GLD")
    parser.add_argument("--target-symbols", default="SPY,QQQ,IWM")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--intraday-range", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--orderbook-assets", default="BTC,ETH,SOL")
    parser.add_argument("--orderbook-date", default="2026-03-06")
    parser.add_argument("--max-train-samples", type=int, default=60_000)
    parser.add_argument("--max-test-samples", type=int, default=30_000)
    parser.add_argument("--sequence-window", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.milestone == "m1":
        results = evaluate_milestone1(n_days=args.n_days, horizon=args.horizon, seed=args.seed)
    elif args.milestone == "m2":
        results = evaluate_milestone2(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            refresh=args.refresh,
        )
    elif args.milestone == "m3":
        context_symbols = tuple(symbol.strip().upper() for symbol in args.context_symbols.split(",") if symbol.strip())
        results = evaluate_milestone3(
            target_symbol=args.symbol,
            context_symbols=context_symbols,
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            refresh=args.refresh,
        )
    elif args.milestone == "m4":
        target_symbols = tuple(symbol.strip().upper() for symbol in args.target_symbols.split(",") if symbol.strip())
        context_symbols = tuple(symbol.strip().upper() for symbol in args.context_symbols.split(",") if symbol.strip())
        context_universe = tuple(dict.fromkeys((args.symbol.upper(), *context_symbols)))
        results = evaluate_milestone4(
            target_symbols=target_symbols,
            context_universe=context_universe,
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            refresh=args.refresh,
        )
    elif args.milestone == "m5":
        target_symbols = tuple(symbol.strip().upper() for symbol in args.target_symbols.split(",") if symbol.strip())
        context_symbols = tuple(symbol.strip().upper() for symbol in args.context_symbols.split(",") if symbol.strip())
        context_universe = tuple(dict.fromkeys((args.symbol.upper(), *context_symbols)))
        results = evaluate_milestone5(
            target_symbols=target_symbols,
            context_universe=context_universe,
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            refresh=args.refresh,
        )
    elif args.milestone == "m6":
        target_symbols = tuple(symbol.strip().upper() for symbol in args.target_symbols.split(",") if symbol.strip())
        results = evaluate_milestone6(
            target_symbols=target_symbols,
            intraday_range=args.intraday_range,
            interval=args.interval,
            horizon=args.horizon,
            refresh=args.refresh,
        )
    elif args.milestone == "m7":
        assets = tuple(asset.strip().upper() for asset in args.orderbook_assets.split(",") if asset.strip())
        results = evaluate_milestone7(
            assets=assets,
            date=args.orderbook_date,
            horizon=args.horizon,
        )
    elif args.milestone == "m8":
        results = evaluate_milestone8(
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    elif args.milestone == "m9":
        results = evaluate_milestone9(
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            sequence_window=args.sequence_window,
            epochs=args.epochs,
        )
    else:
        results = evaluate_milestone10(
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            sequence_window=args.sequence_window,
            epochs=args.epochs,
        )
    write_artifacts(results, args.out_dir / args.milestone)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
