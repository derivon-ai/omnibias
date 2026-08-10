# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch discovery harnesses for self-similar blow-up profiles.

Primary host for CubicGaussNewton / QR Gauss–Newton CCF discovery
(:mod:`omnibias.pinn.torch.discovery.ccf_vorticity_neural`).
"""

from __future__ import annotations

from omnibias.pinn.torch.discovery import ccf_vorticity_neural, multistage

__all__ = [
    "ccf_vorticity_neural",
    "multistage",
]
