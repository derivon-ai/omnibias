# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Battery degradation law-discovery demo.

Recover a compact, interpretable capacity-fade law from battery cycle data
using omnibias closed-form neural operator channels. The synthetic-cycle path
is fully reproducible; the real Severson (2019) path requires downloading the
dataset via ``download_severson.py``.
"""

from __future__ import annotations
