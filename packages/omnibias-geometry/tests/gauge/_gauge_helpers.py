# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic gauge-test inputs: re-export the public BPST helper.

The gold-standard arrays live in
:mod:`omnibias.geometry.gauge._core.instanton` so docs snippets can import
them. This module stays a thin alias so existing ``from _gauge_helpers import
instanton_arrays`` tests keep working.
"""

from __future__ import annotations

from omnibias.geometry.gauge._core.instanton import bpst_instanton_arrays, thooft_eta

instanton_arrays = bpst_instanton_arrays

__all__ = ["instanton_arrays", "thooft_eta"]
