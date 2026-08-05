# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX ``PartitionedField``: bit-parity with the torch twin + ops routing.

The bridge is public (``omnibias.pinn`` is a curated-core package), so its JAX twin
must match the torch field bit-for-bit on identical parameters. Each test builds the
torch and JAX fields from the *same* explicit sub-solution + split parameters and
checks the partition weights, blended forward, and autodiff derivatives agree
(float64, parity ``~1e-8``), plus the fields-ops ``"partitioned"`` routing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("omnibias.partition")  # the optional 'partition' extra (alpha keystone)

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn.jax import ops as jops  # noqa: E402
from omnibias.pinn.jax.fields import OneLayerVectorField as JaxOne  # noqa: E402
from omnibias.pinn.partition.jax import (  # noqa: E402
    PartitionedField as JaxPartitioned,
)
from omnibias.pinn.partition.jax import (  # noqa: E402
    build_partitioned_field as build_jax,
)
from omnibias.pinn.partition.torch import (  # noqa: E402
    PartitionedField as TorchPartitioned,
)
from omnibias.pinn.torch import ops as tops  # noqa: E402
from omnibias.pinn.torch.fields import OneLayerVectorField as TorchOne  # noqa: E402


def _params(rng: np.random.Generator, D: int, H: int, C: int) -> dict[str, np.ndarray]:
    return {
        "W": rng.standard_normal((H, D)),
        "beta": rng.standard_normal(H),
        "c": rng.standard_normal((C, H)),
        "b": rng.standard_normal(C),
    }


def _torch_sub(cs: CoordinateSpec, comp: ComponentSpec, p: dict[str, np.ndarray]) -> TorchOne:
    sub = TorchOne(coordinate_spec=cs, components=comp, hidden=p["W"].shape[0], base="tanh", dtype=torch.float64)
    with torch.no_grad():
        sub.W.weight.copy_(torch.tensor(p["W"], dtype=torch.float64))
        sub.W.bias.copy_(torch.tensor(p["beta"], dtype=torch.float64))
        sub.c.weight.copy_(torch.tensor(p["c"], dtype=torch.float64))
        sub.c.bias.copy_(torch.tensor(p["b"], dtype=torch.float64))
    return sub


def _jax_sub(cs: CoordinateSpec, comp: ComponentSpec, p: dict[str, np.ndarray]) -> JaxOne:
    return JaxOne(
        coordinate_spec=cs,
        components=comp,
        spec=get_activation("tanh"),
        W=jnp.asarray(p["W"]),
        beta=jnp.asarray(p["beta"]),
        c=jnp.asarray(p["c"]),
        b=jnp.asarray(p["b"]),
        hidden=p["W"].shape[0],
    )


def _matched_fields(
    split_dirs: list[list[float]], split_thresh: list[float], *, beta: float = 6.0, H: int = 6, seed: int = 0
) -> tuple[TorchPartitioned, JaxPartitioned]:
    """Torch + JAX PartitionedFields built from *identical* explicit parameters."""
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    rng = np.random.default_rng(seed)
    D = len(split_dirs[0])
    n_regions = 1 << len(split_dirs)
    params = [_params(rng, D, H, 1) for _ in range(n_regions)]
    ft = TorchPartitioned(
        coordinate_spec=cs,
        components=comp,
        subfields=[_torch_sub(cs, comp, p) for p in params],
        split_dirs=torch.tensor(split_dirs, dtype=torch.float64),
        split_thresh=torch.tensor(split_thresh, dtype=torch.float64),
        beta=beta,
        trainable_partition=False,
        dtype=torch.float64,
    )
    fj = JaxPartitioned(
        coordinate_spec=cs,
        components=comp,
        subfields=tuple(_jax_sub(cs, comp, p) for p in params),
        split_W=jnp.asarray(split_dirs, dtype=jnp.float64),
        split_t=jnp.asarray(split_thresh, dtype=jnp.float64),
        depth=len(split_dirs),
        beta=beta,
    )
    return ft, fj


def test_partition_weights_parity() -> None:
    ft, fj = _matched_fields([[1.0]], [0.0], beta=6.0)
    x = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    w_t = ft.partition_weights(torch.tensor(x, dtype=torch.float64)).detach().numpy()
    w_j = np.asarray(fj.partition_weights(jnp.asarray(x)))
    assert w_j.shape == (40, 2)
    assert np.allclose(w_t, w_j, atol=1e-10)
    assert np.allclose(w_j.sum(axis=1), 1.0, atol=1e-12)


def test_forward_values_parity() -> None:
    ft, fj = _matched_fields([[1.0]], [0.0], beta=5.0)
    x = np.random.default_rng(1).standard_normal((13, 1))
    fv_t = ft.forward_values(torch.tensor(x, dtype=torch.float64)).detach().numpy()
    fv_j = np.asarray(fj.forward_values(jnp.asarray(x)))
    assert np.allclose(fv_t, fv_j, atol=1e-10)


def test_derivative_parity_and_finite_difference() -> None:
    ft, fj = _matched_fields([[1.0]], [0.0], beta=4.0)
    x = np.linspace(-1.0, 1.0, 9).reshape(-1, 1)
    xt, xj = torch.tensor(x, dtype=torch.float64), jnp.asarray(x)

    d1_t = tops.derivative(ft(xt), "u", axis=0, order=1).detach().numpy()
    d1_j = np.asarray(jops.derivative(fj(xj), "u", axis=0, order=1))
    assert np.allclose(d1_t, d1_j, atol=1e-8)

    d2_t = tops.derivative(ft(xt), "u", axis=0, order=2).detach().numpy()
    d2_j = np.asarray(jops.derivative(fj(xj), "u", axis=0, order=2))
    assert np.allclose(d2_t, d2_j, atol=1e-7)

    # The JAX autodiff derivative matches an independent central finite difference.
    h = 1e-5
    fp = np.asarray(fj.forward_values(jnp.asarray(x + h)))[:, 0]
    fm = np.asarray(fj.forward_values(jnp.asarray(x - h)))[:, 0]
    assert np.allclose(d1_j, (fp - fm) / (2 * h), atol=1e-6)


def test_heterogeneous_patches_keep_their_parity() -> None:
    """Patches of *different* widths, matched across backends.

    ``_matched_fields`` gives every region the same ``H``; the point of a
    decomposition is that it need not, so the mixed case is what the parity
    claim has to cover.
    """
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    rng = np.random.default_rng(21)
    params = [_params(rng, 1, h, 1) for h in (4, 11)]
    ft = TorchPartitioned(
        coordinate_spec=cs,
        components=comp,
        subfields=[_torch_sub(cs, comp, p) for p in params],
        split_dirs=torch.tensor([[1.0]], dtype=torch.float64),
        split_thresh=torch.tensor([0.0], dtype=torch.float64),
        beta=5.0,
        trainable_partition=False,
        dtype=torch.float64,
    )
    fj = JaxPartitioned(
        coordinate_spec=cs,
        components=comp,
        subfields=tuple(_jax_sub(cs, comp, p) for p in params),
        split_W=jnp.asarray([[1.0]], dtype=jnp.float64),
        split_t=jnp.asarray([0.0], dtype=jnp.float64),
        depth=1,
        beta=5.0,
    )
    assert [sub.hidden for sub in ft.subfields] == [4, 11]

    x = np.linspace(-1.5, 1.5, 17).reshape(-1, 1)
    xt, xj = torch.tensor(x, dtype=torch.float64), jnp.asarray(x)
    assert np.allclose(
        ft.forward_values(xt).detach().numpy(),
        np.asarray(fj.forward_values(xj)),
        atol=1e-10,
    )
    assert np.allclose(
        tops.derivative(ft(xt), "u", axis=0, order=2).detach().numpy(),
        np.asarray(jops.derivative(fj(xj), "u", axis=0, order=2)),
        atol=1e-7,
    )


def test_value_and_gradient_ops_route_partitioned() -> None:
    _, fj = _matched_fields([[1.0]], [0.0])
    x = jnp.asarray(np.random.default_rng(2).standard_normal((5, 1)))
    v = np.asarray(jops.value(fj(x), "u"))
    assert v.shape == (5,)
    g = np.asarray(jops.gradient(fj(x), "u", axes=("x",)))
    assert g.shape == (5, 1)
    assert np.isfinite(g).all()


def test_laplacian_matches_second_derivative_jax() -> None:
    _, fj = _matched_fields([[1.0]], [0.0], beta=4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 7).reshape(-1, 1))
    uxx = np.asarray(jops.derivative(fj(x), "u", axis=0, order=2))
    lap = np.asarray(jops.laplacian(fj(x), "u"))  # 1D spatial laplacian == d2/dx2
    assert np.allclose(uxx, lap, atol=1e-9)


def test_builder_and_depth2() -> None:
    fj = build_jax(
        coordinate_spec=CoordinateSpec(("x",)),
        components=ComponentSpec(("u",)),
        split_dirs=[[1.0], [1.0]],
        split_thresh=[-0.5, 0.5],
        hidden=6,
        beta=8.0,
        seed=0,
    )
    assert fj.n_regions == 4 and len(fj.subfields) == 4
    x = jnp.asarray(np.linspace(-2.0, 2.0, 6).reshape(-1, 1))
    w = np.asarray(fj.partition_weights(x))
    assert w.shape == (6, 4)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_wrong_subfield_count_raises() -> None:
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    sub = _jax_sub(cs, comp, _params(np.random.default_rng(0), 1, 4, 1))
    with pytest.raises(ValueError, match="subfields"):
        JaxPartitioned(
            coordinate_spec=cs,
            components=comp,
            subfields=(sub,),  # depth-1 needs 2
            split_W=jnp.asarray([[1.0]]),
            split_t=jnp.asarray([0.0]),
            depth=1,
        )


def test_pytree_roundtrip_preserves_forward() -> None:
    _, fj = _matched_fields([[1.0]], [0.0])
    leaves, treedef = jax.tree_util.tree_flatten(fj)
    fj2 = jax.tree_util.tree_unflatten(treedef, leaves)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 5).reshape(-1, 1))
    assert np.allclose(
        np.asarray(fj.forward_values(x)), np.asarray(fj2.forward_values(x)), atol=1e-12
    )
