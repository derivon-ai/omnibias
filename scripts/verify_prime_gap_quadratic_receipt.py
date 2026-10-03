#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Replay and seal a coefficient-independent PrimeGaps matrix receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnibias.holonomic.twin_prime_bounded_gap import (  # type: ignore[import-untyped]
    certify_prime_gap_quadratic_receipt,
    seal_prime_gap_quadratic_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    value: object = json.loads(args.receipt.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("receipt root must be a JSON object")
    certificate = certify_prime_gap_quadratic_receipt(
        value,
        dimension=args.dimension,
    )
    sealed: dict[str, object] = seal_prime_gap_quadratic_receipt(certificate)
    rendered = json.dumps(sealed, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    return sealed


if __name__ == "__main__":
    main()
