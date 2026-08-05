# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Spectrum engine (torch) vs exact _core: the jet_partials read-off bridge."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omnibias.boolean._core import anf as _anf  # noqa: E402
from omnibias.boolean._core import multilinear as _ml  # noqa: E402
from omnibias.boolean._core import walsh as _walsh  # noqa: E402
from omnibias.boolean._core.truth_table import pm1_values  # noqa: E402
from omnibias.boolean.torch.ops import spectrum  # noqa: E402


def _tts(seed: int, ns=(1, 2, 3, 4)):  # type: ignore[no-untyped-def]
    rng = random.Random(seed)
    for n in ns:
        for _ in range(8):
            yield tuple(rng.randint(0, 1) for _ in range(1 << n))


def test_mobius_coeffs_match_core() -> None:
    for tt in _tts(10):
        vals = torch.tensor(tt, dtype=torch.float64)
        got = spectrum.mobius_coeffs(vals).detach().numpy()
        want = np.array(_ml.multilinear_coeffs(tt), dtype=np.float64)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_anf_from_jet_partials() -> None:
    for tt in _tts(11):
        vals = torch.tensor(tt, dtype=torch.float64)
        m = spectrum.mobius_coeffs(vals).detach().numpy()
        anf = tuple(int(round(v)) & 1 for v in m)
        assert anf == _anf.anf_from_truth_table(tt)


def test_walsh_coeffs_match_core() -> None:
    for tt in _tts(12):
        vals = torch.tensor(pm1_values(tt), dtype=torch.float64)
        got = spectrum.walsh_coeffs(vals).detach().numpy()
        want = np.array(_walsh.fourier_coeffs(tt, "pm1"), dtype=np.float64)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_influences_match_core() -> None:
    for tt in _tts(13):
        vals = torch.tensor(pm1_values(tt), dtype=torch.float64)
        got = spectrum.influences_diff(vals).detach().numpy()
        want = np.array(_walsh.influences(tt), dtype=np.float64)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_spectrum_is_differentiable() -> None:
    vals = torch.rand(8, dtype=torch.float64, requires_grad=True)
    spectrum.mobius_coeffs(vals).pow(2).sum().backward()
    assert vals.grad is not None


def test_degree_soft_of_parity() -> None:
    # 2-bit XOR is pure order-2 in the Walsh basis -> soft degree == 2.
    xor = (0, 1, 1, 0)
    vals = torch.tensor(pm1_values(xor), dtype=torch.float64)
    assert float(spectrum.algebraic_degree_soft(vals)) == pytest.approx(2.0, abs=1e-9)
