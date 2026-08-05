# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""MNIST-1D double descent: an exact-curvature, optimiser-axis study with omnibias.

Reproduces the model-wise double descent of Greydanus & Kobak
(*Scaling Down Deep Learning with MNIST-1D*, arXiv:2011.14439) and turns it into an
**exact-curvature dissection** (the Hessian spectrum through the interpolation threshold)
and an **optimiser-axis** study (the omnibias second-order suite vs Adam / SGD) that the
SGD/Adam-only literature has not done.

Layout (mirrors ``examples/binary_vs_ste``):

* :mod:`~examples.mnist1d_double_descent.data` -- MNIST-1D (package / vendored / synthetic)
  with fixed per-seed label noise.
* :mod:`~examples.mnist1d_double_descent.models` -- width-parameterised MLP in two
  registers (``ce_relu``, ``mse_tanh``).
* :mod:`~examples.mnist1d_double_descent.arms` -- the optimiser catalogue + closure kinds.
* :mod:`~examples.mnist1d_double_descent.curvature` -- exact dense / matrix-free Hessian
  diagnostics.
* :mod:`~examples.mnist1d_double_descent.train` -- one instrumented run.
* :mod:`~examples.mnist1d_double_descent.experiment` -- the sweep + aggregation.
* :mod:`~examples.mnist1d_double_descent.certify` -- P4 certified read-outs.
* :mod:`~examples.mnist1d_double_descent.run_demo` -- the CLI.
* ``analysis/`` -- figures; ``sweep/`` -- scheduler-neutral fan-out; ``results/`` --
  committed summaries.
"""

from __future__ import annotations

__all__: list[str] = []
