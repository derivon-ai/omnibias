# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Vortex diagnostics (torch-friendly re-exports).

The detector itself runs on NumPy because vortex counting is a pure
post-processing step that does not need autodiff. Users typically::

    phase = torch.atan2(psi_im, psi_re).detach().cpu().numpy()
    n_v, _ = detect_vortices(phase, mask=tf_disk_mask)
"""

from __future__ import annotations

from omnibias.qpinn._core.vortex import (
    VortexDetection,
    detect_vortices,
    detect_vortices_full,
    feynman_vortex_count,
    thomas_fermi_density_2d,
    thomas_fermi_mu_2d,
    thomas_fermi_radius_2d,
)

__all__ = [
    "VortexDetection",
    "detect_vortices",
    "detect_vortices_full",
    "feynman_vortex_count",
    "thomas_fermi_density_2d",
    "thomas_fermi_mu_2d",
    "thomas_fermi_radius_2d",
]
