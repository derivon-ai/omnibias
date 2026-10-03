#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fetch, authenticate, and dimension-parameterize the pinned evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

from omnibias.holonomic.twin_prime_bounded_gap import (  # type: ignore[import-untyped]
    PRIME_GAPS_186_COMMIT,
    PRIME_GAPS_186_SOURCE_SHA256,
    PRIME_GAPS_186_SOURCE_URL,
    build_prime_gap_input_manifest,
    parameterize_prime_gap_evaluator_source,
    prime_gap_engine_layout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=39)
    parser.add_argument("--intervals", type=int, default=98304)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    output = args.output or (
        scratch
        / "twin_prime"
        / f"prime_gap_k{args.dimension}_parameterized.py"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with urlopen(PRIME_GAPS_186_SOURCE_URL, timeout=30) as response:  # noqa: S310
        source_bytes = response.read()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != PRIME_GAPS_186_SOURCE_SHA256:
        raise RuntimeError(
            "pinned PrimeGaps186 source digest changed: "
            f"{source_sha256} != {PRIME_GAPS_186_SOURCE_SHA256}"
        )
    source = source_bytes.decode("utf-8")
    transformed = parameterize_prime_gap_evaluator_source(
        source,
        dimension=args.dimension,
        intervals=args.intervals,
    )
    compile(transformed, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(transformed, encoding="utf-8")
    transformed_sha256 = hashlib.sha256(transformed.encode("utf-8")).hexdigest()
    manifest = build_prime_gap_input_manifest(
        dimension=args.dimension,
        intervals=args.intervals,
    )
    layout = prime_gap_engine_layout(
        dimension=args.dimension,
        intervals=args.intervals,
    )
    receipt: dict[str, object] = {
        "schema": "omnibias.prime_gaps_evaluator_rewrite.v1",
        "upstream_commit": PRIME_GAPS_186_COMMIT,
        "upstream_source_sha256": source_sha256,
        "transformed_source_sha256": transformed_sha256,
        "output": str(output),
        "signed_convolution_strategy": "nonnegative_part_split",
        "resume_protocol": {
            "resume_log": "complete JSON events only",
            "checkpoint": "fsync after every new cap/source event",
            "final_component_requirement": len(manifest.tasks),
        },
        "manifest": manifest.to_payload(),
        "engine_layout": layout.to_payload(),
        "honesty": {
            "source_rewrite_verified": True,
            "generated_source_compiles": True,
            "signed_split_runtime_regression_passed": False,
            "arb_flint_integrals_recomputed": False,
            "finite_numerical_crossing_proved": False,
            "h1_182_claim": False,
        },
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {receipt_path}")
    return receipt


if __name__ == "__main__":
    main()
