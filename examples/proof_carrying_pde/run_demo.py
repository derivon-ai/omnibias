# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run the proof-carrying PDE demo."""

from __future__ import annotations

import json

from examples.proof_carrying_pde.benchmark import evaluate_benchmark


def main() -> None:
    print(json.dumps(evaluate_benchmark(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
