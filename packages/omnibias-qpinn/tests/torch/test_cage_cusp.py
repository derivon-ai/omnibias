# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the nuclear-cusp cage (torch backend).

Twin of :mod:`tests.jax.test_cage_cusp`: autograd parity, the exact
hydrogenic ``-Z^2/2`` oracle, and the Kato ``u'(0) = -Z`` cusp slope.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

torch.set_default_dtype(torch.float64)

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.cage.cusp import (
    NuclearCuspField,
    make_nuclear_cusp_field,
    nuclear_cusp_slope,
)


def _build_base_3d(seed: int = 0) -> OneLayerVectorField:
    torch.manual_seed(seed)
    coord = CoordinateSpec(axes=("x", "y", "z"))
    spec = make_psi_components(name="psi")
    return OneLayerVectorField(
        coordinate_spec=coord, components=spec, hidden=8, base="tanh"
    )


def _cusp_C(r, nuclei, charges, rates):
    tot = r.new_zeros(())
    for a in range(nuclei.shape[0]):
        s = torch.sqrt(torch.sum((r - nuclei[a]) ** 2) + 1e-30)
        tot = tot + (-charges[a] * s / (1.0 + rates[a] * s))
    return torch.exp(tot)


class TestCuspCageConstructor:
    def test_builds_and_keeps_component_names(self):
        base = _build_base_3d()
        cage = make_nuclear_cusp_field(
            base=base, nuclei=torch.zeros((1, 3)), charges=torch.tensor([1.0])
        )
        assert isinstance(cage, NuclearCuspField)
        assert cage.components.names == ("psi_re", "psi_im")

    def test_rejects_wrong_nuclei_dim(self):
        base = _build_base_3d()
        with pytest.raises(ValueError, match="nuclei must be"):
            make_nuclear_cusp_field(
                base=base, nuclei=torch.zeros((1, 2)), charges=torch.tensor([1.0])
            )


class TestCuspCageVsAutograd:
    def test_value_grad_laplacian_mixed_parity(self):
        base = _build_base_3d(seed=1)
        nuclei = torch.tensor([[0.0, 0.0, 0.0], [1.2, -0.3, 0.5]])
        charges = torch.tensor([1.0, 3.0])
        rates = torch.tensor([0.6, 0.9])
        cage = make_nuclear_cusp_field(
            base=base, nuclei=nuclei, charges=charges, rates=rates
        )
        rng = np.random.default_rng(0)
        coords = torch.tensor(rng.normal(size=(4, 3)))
        state = cage(coords)

        def base_val(r, name):
            st = base.evaluate(r[None, :])
            return st.ops.value(st, name)[0]

        for name in ("psi_re", "psi_im"):
            v = state.ops.value(state, name).detach().numpy()
            v_ref = np.array([
                float(
                    (
                        _cusp_C(coords[b], nuclei, charges, rates)
                        * base_val(coords[b], name)
                    ).detach()
                )
                for b in range(4)
            ])
            np.testing.assert_allclose(v, v_ref, atol=1e-11)

            def f(r, name=name):
                return _cusp_C(r, nuclei, charges, rates) * base_val(r, name)

            for ax in range(3):
                d = state.ops.derivative(state, name, axis=ax, order=1).detach().numpy()
                d_ref = np.array([
                    float(torch.autograd.functional.jacobian(f, coords[b])[ax])
                    for b in range(4)
                ])
                np.testing.assert_allclose(d, d_ref, atol=1e-9)

            lap = state.ops.laplacian(state, name).detach().numpy()
            lap_ref = np.array([
                float(torch.trace(torch.autograd.functional.hessian(f, coords[b])))
                for b in range(4)
            ])
            np.testing.assert_allclose(lap, lap_ref, atol=1e-8)

            mp = state.ops.mixed_partial(state, name, (0, 1), (1, 1)).detach().numpy()
            mp_ref = np.array([
                float(torch.autograd.functional.hessian(f, coords[b])[0, 1])
                for b in range(4)
            ])
            np.testing.assert_allclose(mp, mp_ref, atol=1e-8)


class TestHydrogenicOracle:
    @pytest.mark.parametrize("Z", [1.0, 2.0, 3.0])
    def test_constant_base_gives_exact_hydrogenic_energy(self, Z):
        base = _build_base_3d()
        with torch.no_grad():
            base.c.weight.zero_()
            base.c.bias.copy_(torch.tensor([1.0, 0.0]))
        cage = make_nuclear_cusp_field(
            base=base, nuclei=torch.zeros((1, 3)), charges=torch.tensor([Z]), rates=0.0
        )
        rng = np.random.default_rng(int(Z))
        coords = torch.tensor(rng.normal(size=(6, 3)) * 0.7)
        state = cage(coords)
        psi = state.ops.value(state, "psi_re")
        lap = state.ops.laplacian(state, "psi_re")
        s = torch.linalg.norm(coords, dim=-1)
        np.testing.assert_allclose(
            psi.detach().numpy(), torch.exp(-Z * s).detach().numpy(), atol=1e-12
        )
        T_L = -0.5 * lap / psi
        V = -Z / s
        E_L = (T_L + V).detach().numpy()
        np.testing.assert_allclose(E_L, -0.5 * Z * Z, atol=1e-9)


class TestCuspCondition:
    def test_slope_equals_minus_charge(self):
        base = _build_base_3d()
        cage = make_nuclear_cusp_field(
            base=base, nuclei=torch.zeros((2, 3)), charges=torch.tensor([1.0, 6.0])
        )
        assert float(nuclear_cusp_slope(cage, 0)) == -1.0
        assert float(nuclear_cusp_slope(cage, 1)) == -6.0
