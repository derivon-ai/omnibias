# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic JAX discovery harnesses for self-similar blow-up profiles.

Currently ships the Córdoba-Córdoba-Fontelos (CCF) periodic self-similar
discovery / refinement harness (:mod:`omnibias.pinn.jax.discovery.ccf`) plus its
CAP-ready export (:mod:`omnibias.pinn.jax.discovery.cap`).

This subpackage is intentionally *not* auto-imported by
:mod:`omnibias.pinn.jax` -- import it explicitly::

    from omnibias.pinn.jax.discovery import ccf, cap
"""

from __future__ import annotations

from omnibias.pinn.jax.discovery import cap, ccf

__all__ = ["cap", "ccf"]
