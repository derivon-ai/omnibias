# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Use source checkouts for cross-package parity tests."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for path in sorted((Path(__file__).resolve().parents[1] / "packages").glob("*/src")):
    sys.path.insert(0, str(path))
