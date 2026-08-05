# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""End-to-end CCF pipeline: discovery harness -> CAP export -> symbolic check.

Ties the jax discovery harness and CAP exporter (omnibias-pinn) to the
independent numpy validator (omnibias-symbolic). The symbolic package is an
optional dependency, so it is import-skipped when absent.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

sym_ccf = pytest.importorskip(
    "omnibias.symbolic.ccf",
    reason="omnibias-symbolic provides the independent CCF validator",
)

from omnibias.pinn.jax.discovery import cap, ccf  # noqa: E402
from omnibias.pinn.jax.equations.cordoba_cordoba_fontelos import (  # noqa: E402
    ccf_residual_samples,
)


def test_jax_residual_matches_symbolic_numpy_exact_substitution() -> None:
    # The jax operator and the independent numpy reimplementation must agree
    # to ~machine precision on the same samples (exact-substitution check).
    cfg = ccf.CCFDiscoveryConfig(hidden=10, n_grid=192, parity="even", lam_init=0.5)
    theta_star = ccf.default_manufactured_profile()
    _, th, th_y = ccf.manufactured_forcing(cfg, theta_star, 0.5)
    y = ccf.make_grid(cfg)
    jax_res = np.asarray(ccf_residual_samples(y, th, th_y, 0.5))
    np_res = sym_ccf.ccf_self_similar_residual(
        np.asarray(y), np.asarray(th), np.asarray(th_y), 0.5
    )
    np.testing.assert_allclose(jax_res, np_res, atol=1e-10)


def test_symbolic_recovers_lambda_from_manufactured_candidate() -> None:
    cfg = ccf.CCFDiscoveryConfig(hidden=10, n_grid=256, parity="even", lam_init=0.5)
    theta_star = ccf.default_manufactured_profile()
    lam_star = 0.5
    forcing, th, th_y = ccf.manufactured_forcing(cfg, theta_star, lam_star)
    out = sym_ccf.recover_ccf_scaling_law(
        np.asarray(ccf.make_grid(cfg)), np.asarray(th), np.asarray(th_y),
        forcing=np.asarray(forcing),
    )
    assert abs(out["lambda_recovered"] - lam_star) < 1e-8


def test_cap_bundle_independently_verified() -> None:
    cfg = ccf.CCFDiscoveryConfig(hidden=16, n_grid=128, seed=0, lam_init=0.6)
    res = ccf.run_ccf_discovery(cfg, steps=80, lr=5e-3)
    bundle = cap.build_cap_bundle(res, reproduces_published_lambda=None)
    assert cap.cap_schema_errors(bundle) == []
    report = sym_ccf.verify_cap_bundle(bundle)
    # jax-produced residual_samples must match the numpy recomputation.
    assert report["residual_samples_match"], report
    assert report["agreement_max_abs_diff"] < 1e-8
