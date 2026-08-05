# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Back-compat shim for the Phase 1 substrate extraction.

The implementation now lives in :mod:`omnibias.fields.torch.ops.high_order` (package
``omnibias-fields``). This module aliases itself to that module so existing
``omnibias.pinn.torch.ops.high_order`` imports keep working unchanged.
"""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("omnibias.fields.torch.ops.high_order")
