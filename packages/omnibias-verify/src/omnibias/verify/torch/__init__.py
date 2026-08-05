# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Torch frontend for omnibias-verify: ingest a trained ``nn.Module`` into a :class:`~omnibias.verify.Network`, plus closed-form-jet warm-start seeds for the certified minimiser."""

from __future__ import annotations

from omnibias.verify.torch.ingest import network_from_sequential
from omnibias.verify.torch.warm_start import descent_seeds, warm_started_network_minimize

__all__ = [
    "descent_seeds",
    "network_from_sequential",
    "warm_started_network_minimize",
]
