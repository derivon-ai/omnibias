# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scheduler-neutral GPU fan-out for the double-descent sweep.

:mod:`~examples.mnist1d_double_descent.sweep.gen_jobs` emits one self-contained
``run_demo`` command per config; ``submit.sh`` applies a site-supplied submission
wrapper (``$OMNIBIAS_SUBMIT``) to each. No scheduler, queue, host, or absolute path is
hard-coded here -- supply those from your environment.
"""

from __future__ import annotations

__all__: list[str] = []
