# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Heterogeneous per-patch subfields in ``PartitionedField`` (torch + jax).

Domain decomposition is only worth the bookkeeping if the patches can *differ*:
the region holding the boundary layer wants a deep, high-frequency network and
its quiet neighbour does not. These tests pin that the blend accepts mixed field
types and sizes, that every patch still trains, and that the old homogeneous
call keeps working unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omnibias.partition")

from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn.partition.torch import (  # noqa: E402
    PartitionedField,
    build_partitioned_field,
)
from omnibias.pinn.torch import ops  # noqa: E402
from omnibias.pinn.torch.fields import (  # noqa: E402
    JetMLPVectorField,
    OneLayerVectorField,
    build_jet_mlp_vector_field,
)

CS = CoordinateSpec(("x",))
COMPS = ComponentSpec(("u",))
DIRS = [[1.0]]
THRESH = [0.0]


def _x(n: int = 11) -> torch.Tensor:
    return torch.linspace(-1.5, 1.5, n, dtype=torch.float64).reshape(-1, 1)


def _mixed() -> PartitionedField:
    """A deep jet_mlp patch on the left, a cheap one-layer patch on the right."""

    def factory(region: int):
        torch.manual_seed(region)
        if region == 0:
            return build_jet_mlp_vector_field(
                coordinate_spec=CS,
                components=COMPS,
                hidden=12,
                depth=3,
                base="tanh",
                jet_order=2,
                seed=region,
            )
        return OneLayerVectorField(
            coordinate_spec=CS, components=COMPS, hidden=4, base="tanh"
        )

    return build_partitioned_field(
        coordinate_spec=CS,
        components=COMPS,
        split_dirs=torch.tensor(DIRS, dtype=torch.float64),
        split_thresh=torch.tensor(THRESH, dtype=torch.float64),
        beta=6.0,
        subfield_factory=factory,
    )


# -------------------------------------------------------------- torch ---


def test_patches_may_be_different_field_types() -> None:
    field = _mixed()
    assert isinstance(field.subfields[0], JetMLPVectorField)
    assert isinstance(field.subfields[1], OneLayerVectorField)
    x = _x()
    assert field.forward_values(x).shape == (11, 1)


def test_the_blend_is_still_exactly_the_partition_of_unity_sum() -> None:
    field = _mixed()
    x = _x()
    w = field.partition_weights(x)
    with torch.no_grad():
        vals = torch.stack(
            [sub.forward_values(x)[:, 0] for sub in field.subfields], dim=1
        )
        assert torch.allclose(field.forward_values(x)[:, 0], (w * vals).sum(dim=1))


def test_derivatives_of_a_mixed_blend_match_a_finite_difference() -> None:
    """The blend routes through autodiff, mixed patch types and all."""
    field = _mixed()
    x = _x(9)
    ux = ops.derivative(field(x), "u", axis=0, order=1)
    h = 1e-6
    with torch.no_grad():
        fd = (field.forward_values(x + h)[:, 0] - field.forward_values(x - h)[:, 0]) / (
            2 * h
        )
    assert torch.allclose(ux, fd, rtol=1e-6, atol=1e-7)


def test_every_patch_gets_a_gradient() -> None:
    """A patch that never receives one is a patch that is not being trained."""
    field = _mixed()
    ops.derivative(field(_x()), "u", axis=0, order=1).pow(2).mean().backward()
    for i, sub in enumerate(field.subfields):
        grads = [p.grad for p in sub.parameters()]
        assert grads and all(g is not None for g in grads), f"patch {i} got no gradient"
        assert max(float(g.abs().max()) for g in grads) > 0.0


def test_per_region_width_and_activation_without_a_factory() -> None:
    field = build_partitioned_field(
        coordinate_spec=CS,
        components=COMPS,
        split_dirs=torch.tensor(DIRS, dtype=torch.float64),
        split_thresh=torch.tensor(THRESH, dtype=torch.float64),
        hidden=(4, 32),
        base=("tanh", "sin"),
    )
    assert field.subfields[0].hidden == 4
    assert field.subfields[1].hidden == 32
    assert field.subfields[0].spec.name == "tanh"
    assert field.subfields[1].spec.name == "sin"


def test_a_scalar_setting_still_means_every_region_alike() -> None:
    field = build_partitioned_field(
        coordinate_spec=CS,
        components=COMPS,
        split_dirs=torch.tensor([[1.0], [0.5]], dtype=torch.float64),
        split_thresh=torch.tensor([0.0, 0.25], dtype=torch.float64),
        hidden=7,
    )
    assert len(field.subfields) == 4
    assert {sub.hidden for sub in field.subfields} == {7}


@pytest.mark.parametrize("bad", [dict(hidden=(4, 8, 16)), dict(base=("tanh",))])
def test_a_per_region_sequence_of_the_wrong_length_is_rejected(bad) -> None:
    with pytest.raises(ValueError, match="one entry per region"):
        build_partitioned_field(
            coordinate_spec=CS,
            components=COMPS,
            split_dirs=torch.tensor(DIRS, dtype=torch.float64),
            split_thresh=torch.tensor(THRESH, dtype=torch.float64),
            **bad,
        )


def test_patches_must_agree_on_what_they_are_blending() -> None:
    """Silently blending mismatched components would be a wrong answer."""
    good = OneLayerVectorField(coordinate_spec=CS, components=COMPS, hidden=4)
    other = OneLayerVectorField(
        coordinate_spec=CS, components=ComponentSpec(("v",)), hidden=4
    )
    with pytest.raises(ValueError, match="same components"):
        PartitionedField(
            coordinate_spec=CS,
            components=COMPS,
            subfields=[good, other],
            split_dirs=torch.tensor(DIRS, dtype=torch.float64),
            split_thresh=torch.tensor(THRESH, dtype=torch.float64),
        )
    elsewhere = OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("y",)), components=COMPS, hidden=4
    )
    with pytest.raises(ValueError, match="partition is over"):
        PartitionedField(
            coordinate_spec=CS,
            components=COMPS,
            subfields=[good, elsewhere],
            split_dirs=torch.tensor(DIRS, dtype=torch.float64),
            split_thresh=torch.tensor(THRESH, dtype=torch.float64),
        )


def test_the_region_count_is_still_enforced() -> None:
    subs = [OneLayerVectorField(coordinate_spec=CS, components=COMPS, hidden=4)]
    with pytest.raises(ValueError, match="expected 2 subfields"):
        PartitionedField(
            coordinate_spec=CS,
            components=COMPS,
            subfields=subs,
            split_dirs=torch.tensor(DIRS, dtype=torch.float64),
            split_thresh=torch.tensor(THRESH, dtype=torch.float64),
        )


def test_a_skewed_budget_puts_the_parameters_where_it_was_asked_to() -> None:
    """Same total width, concentrated on one side rather than spread evenly.

    Whether that *fits better* is a benchmark question -- it depends on the
    problem, the initialisation and the optimiser, so it is not asserted here.
    What is asserted is the part the library controls: the capacity really does
    move, and the total is unchanged.
    """

    def widths(setting):
        field = build_partitioned_field(
            coordinate_spec=CS,
            components=COMPS,
            split_dirs=torch.tensor(DIRS, dtype=torch.float64),
            split_thresh=torch.tensor(THRESH, dtype=torch.float64),
            hidden=setting,
        )
        return [sub.hidden for sub in field.subfields]

    assert widths(16) == [16, 16]
    assert widths((28, 4)) == [28, 4]


def test_far_from_the_seam_the_blend_is_the_patch_exactly() -> None:
    r"""Why a heterogeneous patch is worth having at all.

    At ``beta = 6`` the gate is saturated by ``x = -6``, so the partition
    weight there is 1 to round-off and the blend *is* the deep patch -- which
    means the patch's exact closed-form ``jet_mlp`` derivative survives into the
    composite to float64 round-off, even though the blend itself is
    differentiated by autodiff. Capacity bought on one side is not diluted on
    that side.
    """
    field = _mixed()
    x = torch.tensor([[-6.0], [-7.0]], dtype=torch.float64)
    w = field.partition_weights(x)[:, 0].detach()
    assert float((1.0 - w).abs().max()) < 1e-14

    deep = field.subfields[0]
    for order in (1, 2):
        blended = ops.derivative(field(x), "u", axis=0, order=order)
        alone = ops.derivative(deep(x), "u", axis=0, order=order)
        assert torch.allclose(blended, alone, rtol=1e-11, atol=1e-12)


# ---------------------------------------------------------------- jax ---


def test_jax_patches_may_be_different_field_types() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.pinn.jax import ops as jops
    from omnibias.pinn.jax.fields import (
        make_jet_mlp_vector_field,
        make_one_layer_vector_field,
    )
    from omnibias.pinn.jax.fields.jet_mlp import JetMLPVectorField as JaxJetField
    from omnibias.pinn.jax.fields.one_layer import (
        OneLayerVectorField as JaxOneLayerField,
    )
    from omnibias.pinn.partition.jax import build_partitioned_field as jax_build

    def factory(region: int):
        if region == 0:
            return make_jet_mlp_vector_field(
                coordinate_spec=CS,
                components=COMPS,
                hidden=12,
                depth=3,
                base="tanh",
                seed=region,
            )
        return make_one_layer_vector_field(
            coordinate_spec=CS, components=COMPS, hidden=4, base="tanh", seed=region
        )

    field = jax_build(
        coordinate_spec=CS,
        components=COMPS,
        split_dirs=[[1.0]],
        split_thresh=[0.0],
        beta=6.0,
        subfield_factory=factory,
    )
    assert isinstance(field.subfields[0], JaxJetField)
    assert isinstance(field.subfields[1], JaxOneLayerField)

    x = jnp.asarray(np.linspace(-1.5, 1.5, 11).reshape(-1, 1))
    w = field.partition_weights(x)
    vals = jnp.stack([sub.forward_values(x)[:, 0] for sub in field.subfields], axis=1)
    assert jnp.allclose(field.forward_values(x)[:, 0], (w * vals).sum(axis=1))

    # The mixed field is still a pytree, so grad reaches both patch types.
    def loss(f):
        return (jops.derivative(f(x), "u", axis=0, order=1) ** 2).mean()

    leaves = jax.tree_util.tree_leaves(jax.grad(loss)(field))
    assert leaves and all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert max(float(jnp.abs(g).max()) for g in leaves) > 0.0


def test_jax_per_region_width_without_a_factory() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from omnibias.pinn.partition.jax import build_partitioned_field as jax_build

    field = jax_build(
        coordinate_spec=CS,
        components=COMPS,
        split_dirs=[[1.0]],
        split_thresh=[0.0],
        hidden=(4, 32),
        base=("tanh", "sin"),
    )
    assert field.subfields[0].W.shape[0] == 4
    assert field.subfields[1].W.shape[0] == 32
    assert field.subfields[0].spec.name == "tanh"
    assert field.subfields[1].spec.name == "sin"


def test_jax_rejects_a_per_region_sequence_of_the_wrong_length() -> None:
    pytest.importorskip("jax")
    from omnibias.pinn.partition.jax import build_partitioned_field as jax_build

    with pytest.raises(ValueError, match="one entry per region"):
        jax_build(
            coordinate_spec=CS,
            components=COMPS,
            split_dirs=[[1.0]],
            split_thresh=[0.0],
            hidden=(4, 8, 16),
        )
