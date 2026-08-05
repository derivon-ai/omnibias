# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Extraction of certified-jet ``(W, b, name)`` layers from a JetMLP-like network."""

from __future__ import annotations

import pytest
from omnibias.core.verified.pde_certificate import (
    certified_interior_residual,
    laplace,
)
from omnibias.verify import (
    certify_pinn_aposteriori,
    verified_layer_bundle,
    verified_layers,
)


class _FakeSpec:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeJetMLP:
    """Minimal stand-in exposing the ``_layer_specs()`` contract (no backend)."""

    def _layer_specs(self) -> list:
        return [
            ([[0.8, -0.5], [0.3, 0.9]], [0.1, -0.2], _FakeSpec("tanh")),
            ([[0.5, 0.2]], [0.0], None),
        ]


def test_backend_neutral_extraction() -> None:
    layers = verified_layers(_FakeJetMLP())
    assert [name for _w, _b, name in layers] == ["tanh", None]
    w0, b0, _ = layers[0]
    assert w0 == [[0.8, -0.5], [0.3, 0.9]] and b0 == [0.1, -0.2]
    # the extracted layers drive the certified jet end-to-end
    r = certified_interior_residual(layers, [(-0.5, 0.5), (-0.5, 0.5)], laplace(2))
    assert r.lo <= r.hi


def test_verified_layer_bundle_records_digest_and_metadata() -> None:
    bundle = verified_layer_bundle(
        _FakeJetMLP(),
        domain=[(-0.5, 0.5), (-0.5, 0.5)],
        provenance={"training": "fixture"},
    )
    assert bundle.metadata["n_layers"] == 2
    assert bundle.metadata["activations"] == ["tanh"]
    assert bundle.metadata["layers_digest"].startswith("sha256:")
    assert bundle.metadata["provenance"]["training"] == "fixture"


def test_string_activation_and_none_bias() -> None:
    class _Net:
        def _layer_specs(self) -> list:
            return [([[1.0, 0.0]], None, "sigmoid"), ([[1.0]], [0.0], None)]

    layers = verified_layers(_Net())
    assert layers[0][1] is None
    assert layers[0][2] == "sigmoid"


@pytest.mark.parametrize("name", ["silu", "gelu", "softplus"])
def test_smooth_neural_activations_supported(name: str) -> None:
    class _Net:
        def _layer_specs(self) -> list:
            return [([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], _FakeSpec(name)), ([[1.0, 1.0]], [0.0], None)]

    layers = verified_layers(_Net())
    assert layers[0][2] == name
    # the extracted layers drive the certified jet end-to-end
    r = certified_interior_residual(layers, [(-0.3, 0.3), (-0.3, 0.3)], laplace(2))
    assert r.lo <= r.hi


def test_unsupported_activation_raises() -> None:
    class _Siren:
        def _layer_specs(self) -> list:
            return [([[1.0]], [0.0], _FakeSpec("sin"))]

    with pytest.raises(ValueError, match="not supported"):
        verified_layers(_Siren())


def test_missing_layer_specs_raises() -> None:
    with pytest.raises(TypeError, match="_layer_specs"):
        verified_layers(object())


def test_torch_jetmlp_end_to_end() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.torch.architectures.pinn import JetMLP

    torch.manual_seed(0)
    net = JetMLP(in_dim=2, hidden=4, out_dim=1, depth=2, base="tanh").double()
    layers = verified_layers(net)

    def forward(p: torch.Tensor) -> torch.Tensor:
        z = p
        for w, b, spec in net._layer_specs():
            z = z @ w.T
            if b is not None:
                z = z + b
            if spec is not None:
                z = torch.tanh(z)
        return z.squeeze(-1)

    for px, py in [(0.2, -0.3), (-0.5, 0.4)]:
        p = torch.tensor([px, py], dtype=torch.float64, requires_grad=True)
        grad = torch.autograd.grad(forward(p), p, create_graph=True)[0]
        lap = sum(
            torch.autograd.grad(grad[i], p, retain_graph=True)[0][i] for i in range(2)
        )
        r = certified_interior_residual(layers, [(px, px), (py, py)], laplace(2))
        assert r.lo - 1e-9 <= float(lap) <= r.hi + 1e-9


def test_certify_pinn_aposteriori_from_torch_jetmlp() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.torch.architectures.pinn import JetMLP

    torch.manual_seed(0)
    net = JetMLP(in_dim=2, hidden=3, out_dim=1, depth=1, base="tanh").double()
    result = certify_pinn_aposteriori(
        net,
        [(-0.1, 0.1), (-0.1, 0.1)],
        laplace(2),
        target_residual=None,
        max_splits=1,
        provenance={"seed": 0},
    )
    assert result.certificate.certificate["payload"]["model"]["layers_digest"].startswith("sha256:")
    assert result.diagnostics is not None
    assert result.diagnostics.boxes == 1
